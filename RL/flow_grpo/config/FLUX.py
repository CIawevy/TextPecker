
import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    #vllm server config
    config.vllm_host = "2605:340:cd60:0:9e1c:a6fd:1ee2:d01b" #replace with your own
    config.vllm_port = 8848
    # flux

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True

    return config



    

def get_config(name):
    return globals()[name]()

def general_ocr_flux_fast_guard_vkl_beta():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr_v2_en")
    #Flux.1[dev]
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 28
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5
    config.sample.eval_guidance_scale = 3.5
    #fast setting
    config.sample.sde_window_size = 9
    config.sample.sde_window_range = (0, config.sample.num_steps//2) 
    config.sample.sde_type = "sde"
    config.sample.noise_level = 0.9
    # config.train.timestep_fraction = 0.99 #windows_size=0时效果一致

    config.run_name =  "base_vkl_fastguard_beta4_st28_w9_cfg_clip0.02_lr1_n0.9"
    config.run_project = "FLOW-RL-Flux"

    #vllm server config
    config.vllm_host = "2605:340:cd60:0:9e1c:a6fd:1ee2:d01b" #replace with your own
    config.vllm_port = 8848
    

    config.resolution = 512
    config.sample.train_batch_size = 3 #每张卡每次负责3个prompt

    #OOM 
    config.sample.num_image_per_prompt = 24 #每个prompt 要24个图
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt)) # 48
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    
    config.train.vkl = True
    config.train.beta = 4e-4 #4e-5 hacking
    config.sample.global_std = False
    config.sample.same_latent = False
    config.train.ema = True
    config.mixed_precision = "fp16"
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = f'checkpoints/logs/ocr_v2_en/flux/{config.run_name}'
    config.reward_fn = {
        "ocr_en": 1.0,
        # "pickscore": 0.0
    }
    config.eval_reward_fn = {
        "textpecker_score_vllm_eval": {'semantic': 0.5, 'quality': 0.5},
        # 'ocr_en': 1.0,
        # "pickscore": 0.0
    }
    
    config.prompt_fn = "general_ocr"
    #GRPO-GUARD
    config.train.clip_range=2e-6 #1e-5
    config.train.learning_rate = 1e-4
    config.rationorm = True

    config.per_prompt_stat_tracking = True
    return config
def general_ocr_flux_fast_guard_vkl_beta_mropa():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr_v2_en")
    #Flux.1[dev]
    config.pretrained.model =  "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 28
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5
    config.sample.eval_guidance_scale = 3.5
    #fast setting
    config.sample.sde_window_size = 9
    config.sample.sde_window_range = (0, config.sample.num_steps//2) 
    config.sample.sde_type = "sde"
    config.sample.noise_level = 0.9
    # config.train.timestep_fraction = 0.99 #windows_size=0时效果一致

    config.run_name =  "mropa712_vkl_fastguard_beta1_st28_w9_cfg_clip0.02_lr1_n0.9"
    config.run_project = "FLOW-RL-Flux"

    #vllm server config
    config.vllm_host = "2605:340:cd60:0:9e1c:a6fd:1ee2:d01b" #replace with your own
    config.vllm_port = 8848
    

    config.resolution = 512
    config.sample.train_batch_size = 3 #每张卡每次负责3个prompt

    #OOM 
    config.sample.num_image_per_prompt = 24 #每个prompt 要24个图
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt)) # 48
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    
    config.train.vkl = True
    config.train.beta = 1e-4 #4e-5 hacking
    config.sample.global_std = False
    config.sample.same_latent = False
    config.train.ema = True
    config.mixed_precision = "fp16"
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = f'checkpoints/logs/ocr_v2_en/flux/{config.run_name}'
    config.reward_fn = {
        'ocr_en': 0.7,
        "pickscore": 0.1,
        "aesthetic": 0.2,
    }
    config.eval_reward_fn = {
        "textpecker_score_vllm_eval": {'semantic': 0.5, 'quality': 0.5},
        # 'ocr_en': 1.0,
        # "pickscore": 0.0
    }
    
    config.prompt_fn = "general_ocr"
    #GRPO-GUARD
    config.train.clip_range=2e-6 #1e-5
    config.train.learning_rate = 1e-4
    config.rationorm = True

    config.per_prompt_stat_tracking = True
    return config

def general_ocr_flux_fast_guard_vkl_pecker_mrtpa():
    #flux 的best config 调参差不多了，clip 和 lr 保证了 clip frac 0.1-0.2 ，现在就是beta如何利于学习又防止hacking
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr_v2_en")
    #Flux.1[dev]
    config.pretrained.model =  "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 28
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5
    config.sample.eval_guidance_scale = 3.5
    #fast setting
    config.sample.sde_window_size = 9
    config.sample.sde_window_range = (0, config.sample.num_steps//2) 
    config.sample.sde_type = "sde"
    config.sample.noise_level = 0.9
    # config.train.timestep_fraction = 0.99 #windows_size=0时效果一致

    config.run_name =  "mrTpa5212_vkl_fastguard_beta1_st28_w9_cfg_clip0.02_lr1_n0.9"
    config.run_project = "FLOW-RL-Flux"

    #vllm server config
    config.vllm_host = "2605:340:cd60:0:9e1c:a6fd:1ee2:d01b" #replace with your own
    config.vllm_port = 8848
    

    config.resolution = 512
    config.sample.train_batch_size = 3 #每张卡每次负责3个prompt

    #OOM 
    config.sample.num_image_per_prompt = 24 #每个prompt 要24个图
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt)) # 48
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    
    config.train.vkl = True
    config.train.beta = 1e-4 #4e-5 hacking
    config.sample.global_std = False
    config.sample.same_latent = False
    config.train.ema = True
    config.mixed_precision = "fp16"
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = f'checkpoints/logs/ocr_v2_en/flux/{config.run_name}'
    config.reward_fn = {
        "textpecker_score_vllm": {'semantic': 0.5, 'quality': 0.2},
        # 'ocr_en': 0.8,
        "pickscore": 0.1,
        "aesthetic": 0.2,
    }
    config.eval_reward_fn = {
        "textpecker_score_vllm_eval": {'semantic': 0.5, 'quality': 0.5},
        # "pickscore": 0.2
        # 'ocr_en': 1.0,
        # "pickscore": 0.0
    }
    
    
    config.prompt_fn = "general_ocr"
    #GRPO-GUARD
    config.train.clip_range=2e-6 #1e-5
    config.train.learning_rate = 1e-4
    config.rationorm = True

    config.per_prompt_stat_tracking = True
    return config

