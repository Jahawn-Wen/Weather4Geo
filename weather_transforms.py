import imgaug.augmenters as iaa

def build_iaa_transform(weather, h, w):
    aug_list = []

    if weather in ['normal', 'none']:
        # 不做显式天气增强，只做 resize
        pass
    elif weather == 'dark':
        # 原脚本中 dark 的典型配置
        aug_list += [
            iaa.BlendAlpha(0.5, foreground=iaa.Add(100), background=iaa.Multiply(0.2), seed=31),
            iaa.MultiplyAndAddToBrightness(mul=0.2, add=(-30, -15), seed=1991),
        ]
    elif weather == 'rain':
        aug_list += [
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),
        ]
    elif weather == 'snow':
        aug_list += [
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96),
        ]
    elif weather == 'fog':
        aug_list += [
            iaa.CloudLayer(
                intensity_mean=225, intensity_freq_exponent=-2,
                intensity_coarse_scale=2, alpha_min=1.0,
                alpha_multiplier=0.9, alpha_size_px_max=10,
                alpha_freq_exponent=-2, sparsity=0.9,
                density_multiplier=0.5, seed=35
            ),
        ]
    elif weather == 'fog_rain':
        aug_list += [
            iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
                           alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
                           density_multiplier=0.5, seed=35),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36),
        ]
    elif weather == 'fog_snow':
        aug_list += [
            iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2, intensity_coarse_scale=2, alpha_min=1.0,
                           alpha_multiplier=0.9, alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
                           density_multiplier=0.5, seed=35),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36),
        ]
    elif weather == 'rain_snow':
        aug_list += [
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
        ]
    elif weather == 'light':
        aug_list += [
            iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992),
        ]
    elif weather == 'wind':
        aug_list += [
            iaa.MotionBlur(15, seed=17),
        ]
    else:
        print(f"[WARN] Unknown weather '{weather}', fallback to no weather augmentation.")

    # 所有 weather 最终都要 resize 到 (h, w)
    aug_list.append(iaa.Resize({"height": h, "width": w}, interpolation=3))
    return iaa.Sequential(aug_list)

