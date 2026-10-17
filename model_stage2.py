"""75.85 model: active X-VLM branch only; state-dict names are retained."""
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from transformers import BertTokenizer, RobertaTokenizer
from runtime_paths import configure_imports
configure_imports()
from models.xvlm import XVLMBase

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class WeatherAligner(nn.Module):
    """
    一个很小的残差适配模块：
        f_align = f_img + Δ(f_img, w_embed)
    其中 w_embed 是天气 embedding（one-hot / learnable embedding）。
    """
    def __init__(self, feat_dim, num_weathers, emb_dim=64):
        super().__init__()
        self.weather_embed = nn.Embedding(num_weathers, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + emb_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, f_img, w_ids):
        """
        f_img:  [B, D]
        w_ids:  [B]   天气 ID，0 通常表示 normal，其余表示不同天气
        """
        w = self.weather_embed(w_ids)           # [B, emb_dim]
        x = torch.cat([f_img, w], dim=-1)       # [B, D + emb_dim]
        delta = self.mlp(x)                     # [B, D]
        return f_img + delta


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, num_bottleneck=512, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        self.Linear = nn.Linear(input_dim, num_bottleneck)
        self.bnorm = nn.BatchNorm1d(num_bottleneck)
        self.dropout = nn.Dropout(p=droprate)
        init.kaiming_normal_(self.Linear.weight.data, a=0, mode='fan_out')
        init.constant_(self.Linear.bias.data, 0.0)
        init.normal_(self.bnorm.weight.data, 1.0, 0.02)
        init.constant_(self.bnorm.bias.data, 0.0)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.Linear(x)
        x = self.bnorm(x)
        x = self.dropout(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x,f]
        else:
            x = self.classifier(x)
            return x


class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg',
                 share_weight = False, VGG16=False, LPN=False, block=6,
                 norm='bn', adain='a', circle=False, btnk=[1,0,1],
                 conv_norm='none', config_xvlm=None,
                 num_weathers=None, proto_m=0.01):
        """
        Stage2_3 版本 three_view_net：
          - Stage1：整体对齐 (CE + ITC/ITM)，学习一个较好的 backbone 表征；
          - Stage2：冻结 backbone，仅训练 WeatherAligner，
            通过 residual 的方式对齐跨天气特征到 normal 天气的 canonical 表征。
        """
        super(three_view_net, self).__init__()
        self.LPN   = LPN
        self.block = block
        self.norm  = norm
        self.adain = adain
        self.circle = circle

        assert config_xvlm is not None
        self.xvlm = XVLMBase(
            config=config_xvlm,
            load_vision_params=config_xvlm.get("load_vision_pretrained", True),
            load_text_params=True,
            use_contrastive_loss=True,
            use_matching_loss=True,
            use_mlm_loss=config_xvlm.get('use_mlm_loss', True),
            use_bbox_loss=config_xvlm.get('use_bbox_loss', True)
        )
        self.xvlm.config   = config_xvlm
        self.vision_encoder = self.xvlm.vision_encoder
        self.text_encoder   = self.xvlm.text_encoder
        self.vision_proj    = self.xvlm.vision_proj
        self.text_proj      = self.xvlm.text_proj
        self.temp           = self.xvlm.temp
        self.itm_head       = self.xvlm.itm_head
        if config_xvlm.get('use_roberta', False):
            self.tokenizer = RobertaTokenizer.from_pretrained(config_xvlm['text_encoder'])
        else:
            self.tokenizer = BertTokenizer.from_pretrained(config_xvlm['text_encoder'])

        in_feat = config_xvlm['embed_dim']
        self.classifier = ClassBlock(in_feat, class_num, droprate, return_f=circle)

        # WeatherAligner + Prototype bank：二者并存
        self.num_classes  = class_num
        self.num_weathers = num_weathers  # 由外部传入 = len(weather_names)+1
        self.proto_m      = proto_m
        if self.num_weathers is None:
            # 退化为 1 个天气（不区分），基本等价于不做天气对齐
            self.num_weathers = 1

        # 文本原型：每个 (ID, weather) 一个 prototype
        self.text_prototypes = None  # [C, W, D]
        self.proto_counts    = None  # [C, W]

        # WeatherAligner：只在 Stage2 训练，用于跨天气对齐
        self.weather_aligner = WeatherAligner(in_feat, self.num_weathers, emb_dim=64)

        # 阶段开关：默认只做 Stage1（整体对齐），到指定 epoch 后由训练脚本打开 Stage2
        self.stage2_on = False

        # 用于缓存当前 batch 的 loss
        self.loss_itc        = None
        self.loss_itm        = None
        self.loss_align      = None   # 图像对齐 (f_align vs f_normal)
        self.loss_align_text = None   # 文本原型对齐 (f_align vs t_proto_normal)

        # Stage1 默认不训练 WeatherAligner
        for p in self.weather_aligner.parameters():
            p.requires_grad = False

    def freeze_backbone(self):
        """
        进入 Stage2 时调用：
        - 冻结除 WeatherAligner 外的全部参数；
        - 只训练 WeatherAligner。
        """
        for name, p in self.named_parameters():
            p.requires_grad = False
        for p in self.weather_aligner.parameters():
            p.requires_grad = True

    def _init_prototypes(self, device, dim):
        """初始化文本 prototype bank"""
        C = self.num_classes
        W = self.num_weathers if self.num_weathers is not None else 1
        self.text_prototypes = torch.zeros(C, W, dim, device=device)
        self.proto_counts    = torch.zeros(C, W, device=device)

    def _encode_image_feat(self, img):
        """使用 X-VLM 提取图像 CLS 向量，并投影到联合空间"""
        img_embeds, img_atts = self.xvlm.get_vision_embeds(img)   # [B, 1+P, C]
        img_cls = img_embeds[:, 0, :]                              # [B, C]
        img_feat = F.normalize(self.vision_proj(img_cls), dim=-1)  # [B, D]
        return img_feat, img_embeds, img_atts

    def forward(self, x1, x2, x3, x4=None,
                captions=None,
                cls_indices=None,
                weather_indices=None,
                scene_captions=None,
                weather_captions=None):
        """
        x1, x2, x3, x4 : satellite / street / drone / google views
        captions       : List[str]，当前 batch 的整句 caption
        cls_indices    : Tensor[int64]，drone 对应的 ID（labels3）
        weather_indices: Tensor[int64]，drone 对应的天气标签（wlabels3），
                         约定 0=normal，其余为各种天气增强。
        其余 scene_captions / weather_captions 参数只为兼容旧接口，这里不使用。
        """
        # reset loss cache
        self.loss_itc        = None
        self.loss_itm        = None
        self.loss_align      = None
        self.loss_align_text = None

        # 选一个已有图像 tensor 确定 device
        any_img = next((t for t in (x1, x2, x3, x4) if t is not None), None)
        if any_img is not None:
            device = any_img.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        text_feat  = None
        txt_embeds = None
        cap_inputs = None

        # —— 1) 编码整句 caption & 更新文本原型 —— #
        if captions is not None and self.training:
            cap_inputs = self.tokenizer(
                captions,
                padding='longest',
                truncation=True,
                max_length=self.xvlm.config['max_tokens'],
                return_tensors="pt"
            ).to(device)
            txt_out = self.text_encoder(
                cap_inputs.input_ids,
                attention_mask=cap_inputs.attention_mask,
                return_dict=True,
                mode='text',
                output_hidden_states=True
            )
            txt_embeds = txt_out.hidden_states[-1]                # [B, T, H]
            text_feat  = F.normalize(self.text_proj(txt_embeds[:, 0, :]), dim=-1)  # [B, D]

            # 在线 EMA 更新文本 prototype：per-(ID, weather)
            if cls_indices is not None and weather_indices is not None:
                if self.text_prototypes is None:
                    self._init_prototypes(device, text_feat.size(-1))
                with torch.no_grad():
                    B = text_feat.size(0)
                    for i in range(B):
                        c = int(cls_indices[i])
                        w = int(weather_indices[i])
                        if c < 0 or c >= self.num_classes:
                            continue
                        if w < 0 or w >= self.num_weathers:
                            continue
                        cur   = text_feat[i].detach()
                        count = self.proto_counts[c, w]
                        if count == 0:
                            new = cur
                        else:
                            m   = self.proto_m
                            new = m * self.text_prototypes[c, w] + (1.0 - m) * cur
                        new = F.normalize(new, dim=0)
                        self.text_prototypes[c, w] = new
                        self.proto_counts[c, w]    = count + 1

        # —— 2) helper：跑一张图像 —— #
        def run_one(img, view_name=None, use_text_loss: bool = False,
                    cls_idx=None, weather_idx=None):
            img_feat, img_embeds, img_atts = self._encode_image_feat(img)  # [B, D]

            # 默认使用 raw 特征
            feat_for_loss = img_feat

            # Stage2 + drone 分支：对天气样本走 WeatherAligner
            if (self.training and self.stage2_on and
                view_name == 'drone' and weather_idx is not None):
                w_flat = weather_idx.view(-1)
                feat_for_loss = img_feat.clone()
                # w=0 视为 normal，不做调整；w>0 且 < num_weathers 视为天气增强
                mask_weather = (w_flat > 0) & (w_flat < self.num_weathers)
                if mask_weather.any():
                    feat_for_loss[mask_weather] = self.weather_aligner(
                        img_feat[mask_weather],
                        w_flat[mask_weather]
                    )

                # 计算跨天气图像对齐损失：f_align(weather) → f_normal(raw)
                self.loss_align = None
                self.loss_align_text = None
                if cls_idx is not None:
                    ids   = cls_idx.view(-1)
                    f_raw = img_feat.detach()   # normal canonical，不回传梯度
                    f_aln = feat_for_loss       # weather 对齐后的特征

                    # 图像对齐：同 ID 的 normal vs weather
                    img_loss_terms = []
                    B = f_raw.size(0)
                    for i in range(B):
                        # 只把当前样本当作 normal canonical
                        if w_flat[i].item() != 0:
                            continue
                        pid = ids[i]
                        # 同一 ID 且天气 != normal 的样本
                        mask_pair = (ids == pid) & (w_flat != 0)
                        if mask_pair.any():
                            f_normal  = f_raw[i]              # [D]
                            f_weather = f_aln[mask_pair]      # [N_w, D]
                            diff = f_weather - f_normal.unsqueeze(0)
                            img_loss_terms.append(diff.pow(2).sum(dim=1))  # ||·||^2
                    if img_loss_terms:
                        self.loss_align = torch.cat(img_loss_terms, dim=0).mean()

                    # 文本原型对齐：f_align(weather) → text_proto(id, normal)
                    if self.text_prototypes is not None:
                        text_loss_terms = []
                        for i in range(B):
                            # 只对 weather 样本做文本对齐
                            if w_flat[i].item() == 0:
                                continue
                            pid = int(ids[i])
                            if pid < 0 or pid >= self.num_classes:
                                continue
                            # normal weather 的原型默认索引 0
                            if 0 < 0 or 0 >= self.num_weathers:
                                continue
                            if self.proto_counts is None or self.proto_counts[pid, 0] <= 0:
                                continue
                            t_proto = self.text_prototypes[pid, 0]         # [D]
                            t_proto = F.normalize(t_proto, dim=0)
                            cos = (f_aln[i] * t_proto.to(f_aln.device)).sum()
                            text_loss_terms.append(-cos)
                        if text_loss_terms:
                            self.loss_align_text = torch.stack(text_loss_terms).mean()

            # ITC / ITM：两个阶段都用，保持整体图文对齐
            if self.training and use_text_loss and text_feat is not None:
                loss_itc = self.xvlm.get_contrastive_loss(feat_for_loss, text_feat)
                loss_itm = self.xvlm.get_matching_loss(
                    img_embeds, img_atts,
                    feat_for_loss,
                    txt_embeds, cap_inputs.attention_mask,
                    text_feat
                )
                self.loss_itc = loss_itc
                self.loss_itm = loss_itm

            # 分类：所有 view 都用 feat_for_loss 做 ID 分类
            out = self.classifier(feat_for_loss)
            return out

        # —— 3) 分别处理三路（可选四路） —— #
        y1 = None if x1 is None else run_one(x1, view_name='sat',    use_text_loss=False)
        y2 = None if x2 is None else run_one(x2, view_name='street', use_text_loss=False)
        y3 = None
        if x3 is not None:
            y3 = run_one(
                x3, view_name='drone',
                use_text_loss=(captions is not None),
                cls_idx=cls_indices, weather_idx=weather_indices
            )

        if x4 is None:
            return y1, y2, y3
        else:
            y4 = run_one(x4, view_name='google', use_text_loss=False)
            return y1, y2, y3, y4

    def part_classifier(self, x):
        part = {}
        predict = {}
        for i in range(self.block):
            part[i] = x[:,:,i].view(x.size(0),-1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y

