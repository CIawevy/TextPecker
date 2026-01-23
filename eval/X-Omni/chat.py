import torch
import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


def main(args):
    torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).cuda()
    model.init_vision(args.flux_model_name_or_path)
    model.set_generation_mode('text')
    model.eval()

    image = Image.open(args.image_path).convert('RGB')
    image_str = model.tokenize_image(image)
    message = [{'role': 'user', 'content': image_str + '\n' + args.prompt}]
    input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors='pt')
    with torch.inference_mode():
        generation_config = GenerationConfig(
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
            top_p=None,
            num_beams=1,
            use_cache=True, 
            eos_token_id=tokenizer.encode('<|im_end|>')[0],
            pad_token_id=0,
        )
        output_ids = model.generate(input_ids.cuda(), generation_config=generation_config)
        texts, _ = model.mmdecode(tokenizer, output_ids[:, input_ids.shape[1]: -1])
    print('Question: ', args.prompt)
    print('Response: ', texts[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--flux_model_name_or_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
    parser.add_argument("--image-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
