import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"
    config.train.vkl = False

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config


def get_config(name):
    return globals()[name]()





def general_ocr_sd3_fastguard_vkl_beta():
    #fast guard + vkl regularization
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr_v2_en")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5 #no cfg for old policy sampling
    config.sample.noise_level = 0.9
    config.sample.eval_guidance_scale = 4.5 #cfg for eval
    config.train.cfg = True #no cfg for new policy sampling 

    config.run_name =  "base_vkl_fastguard_beta4_st40_w12_cfg_clip0.02_lr1_n0.9"
    config.run_project = "FLOW-RL-SD"

    #vllm server config
    config.vllm_host = "2605:340:cd60:0:9e1c:a6fd:1ee2:d01b" #replace with your own
    config.vllm_port = 8848
    
    #fast setting
    config.sample.sde_window_size = 12 #>0 ensure rationorm=False  # 15 * 3/5 = 9
    config.sample.sde_window_range = (0, config.sample.num_steps//2) #15
    config.sample.sde_type = "sde"

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # config.train.timestep_fraction = 0.99 #避免最后一步 与sde窗口冲突
    
    # kl loss
    config.train.beta = 4e-4  #4e-5 still hacking latter
    config.train.vkl = True
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = False #!!! for guard
    # Whether to use the same noise for the same prompt
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = f'checkpoints/logs/ocr_v2_en/sd3.5/{config.run_name}'
    config.reward_fn = {
        'ocr_en': 1.0,
        # "pickscore": 0.0
    }
    config.eval_reward_fn = {
        "textpecker_score_vllm_eval": {'semantic': 0.5, 'quality': 0.5},
        # 'ocr_en': 1.0,
        # "pickscore": 0.0
    }
    
    config.prompt_fn = "general_ocr"
    config.mixed_precision = "fp16"

    config.train.clip_range=2e-6 #
    config.train.learning_rate = 1e-4 #default
    config.rationorm = True 

    config.per_prompt_stat_tracking = True
    return config




def general_ocr_sd_fast_guard_vkl_beta_mropa():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr_v2_en")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5 #no cfg for old policy sampling
    config.sample.noise_level = 0.9
    config.sample.eval_guidance_scale = 4.5 #cfg for eval
    config.train.cfg = True #no cfg for new policy sampling 

    config.run_name =  "mr_opa712_vkl_fastguard_beta1_st40_w12_cfg_clip0.02_lr1_n0.9"
    config.run_project = "FLOW-RL-SD"

    #vllm server config
    config.vllm_host = "2605:340:cd60:0:9e1c:a6fd:1ee2:d01b" #replace with your own
    config.vllm_port = 8848
    
    #fast setting
    config.sample.sde_window_size = 12 #>0 ensure rationorm=False  # 15 * 3/5 = 9
    config.sample.sde_window_range = (0, config.sample.num_steps//2) #15
    config.sample.sde_type = "sde"

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # config.train.timestep_fraction = 0.99 #避免最后一步 与sde窗口冲突
    
    # kl loss
    config.train.beta = 1e-4  #4e-5 still hacking latterE
    config.train.vkl = True
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = False #!!! for guard
    # Whether to use the same noise for the same prompt
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = f'checkpoints/logs/ocr_v2_en/sd3.5/{config.run_name}'
    config.reward_fn = {
        'ocr_en': 0.7,
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
    config.mixed_precision = "fp16"

    config.train.clip_range=2e-6 #
    config.train.learning_rate = 1e-4 #default
    config.rationorm = True 

    config.per_prompt_stat_tracking = True
    return config





def general_ocr_sd_fast_guard_vkl_beta_mrtpa():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr_v2_en")

    # sd3.5 medium
    config.pretrained.model =  "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5 #no cfg for old policy sampling
    config.sample.noise_level = 0.9
    config.sample.eval_guidance_scale = 4.5 #cfg for eval
    config.train.cfg = True #no cfg for new policy sampling 

    config.run_name =  "mr_tpa5212_vkl_fastguard_beta1_st40_w12_cfg_clip0.02_lr1_n0.9"
    config.run_project = "FLOW-RL-SD"

    #vllm server config
    config.vllm_host = "2605:340:cd60:0:9e1c:a6fd:1ee2:d01b" #replace with your own
    config.vllm_port = 8848
    
    #fast setting
    config.sample.sde_window_size = 12 #>0 ensure rationorm=False  # 15 * 3/5 = 9
    config.sample.sde_window_range = (0, config.sample.num_steps//2) #15
    config.sample.sde_type = "sde"

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))

    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # config.train.timestep_fraction = 0.99 #避免最后一步 与sde窗口冲突
    
    # kl loss
    config.train.beta = 1e-4  #4e-5 still hacking latterE
    config.train.vkl = True
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = False #!!! for guard
    # Whether to use the same noise for the same prompt
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = f'checkpoints/logs/ocr_v2_en/sd3.5/{config.run_name}'
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
    config.mixed_precision = "fp16"

    config.train.clip_range=2e-6 #
    config.train.learning_rate = 1e-4 #default
    config.rationorm = True 

    config.per_prompt_stat_tracking = True
    return config