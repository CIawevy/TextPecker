
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

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def get_config(name):
    return globals()[name]()


def general_ocr_qwen_fast_guard_vkl_beta_mrtpa():
    #beta 4显然不会hack 目前setting clip非常好
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr_v2")
    #Flux.1[dev]
    config.pretrained.model = "Qwen/Qwen-Image"
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4
    config.sample.eval_guidance_scale = 4
    config.sample.noise_level = 1.2
    #fast setting
    config.sample.sde_window_size = 5
    config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.sample.sde_type = "sde"

    config.run_name = "mrTpa5212_vkl_fastguard_beta10_w5_st20_clip0.2_lr1_n1.2" #replace with your own
    config.run_project = "FLOW-RL-Qwen"

    #vllm server config
    config.vllm_host = "2605:340:cd60:0:9e1c:a6fd:1ee2:d01b"
    config.vllm_port = 8850
    

    config.resolution = 512
    config.sample.train_batch_size = 4
    
    #OOM 
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(32/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 4 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.


    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.beta = 1e-3 #!!! 4e-5 directly hacking， for its kl loss is smaller than orthers
    config.train.vkl = True
    config.sample.global_std = False #!!!!refine and close
    config.sample.same_latent = False
    
    config.train.ema = False #!!!!default false
    config.mixed_precision = "bf16"
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = f'checkpoints/logs/ocr_v2/qwenimage/{config.run_name}'
    config.reward_fn = {
        "textpecker_score_vllm": {'semantic': 0.5, 'quality': 0.2},
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
    config.train.clip_range=2e-5
    config.train.learning_rate = 1e-4
    config.rationorm = True

    config.per_prompt_stat_tracking = True
    return config

