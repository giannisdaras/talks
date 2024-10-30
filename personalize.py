from utils import load_image, save_image, blur_image
import argparse
import torch
import os
parser = argparse.ArgumentParser("Parser for talk personalization.")
parser.add_argument("--host_image", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--to_blur_image", type=str, required=True)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    host_image = load_image(args.host_image, device='cpu') * 2 - 1
    host_image_name = os.path.basename(args.host_image).split('.')[0]
    noise = torch.randn_like(host_image)
    for noise_level in [0.5, 1.0, 2.0, 4.0, 5.0]:
        noisy_image = host_image + noise * noise_level
        save_image(noisy_image.squeeze(0), os.path.join(args.output_dir, f"{host_image_name}_noise_{noise_level}.png"))
    
    blurred_image = blur_image(load_image(args.to_blur_image, device='cpu')[0] * 2 - 1, 5.0)
    to_blur_image_name = os.path.basename(args.to_blur_image).split('.')[0]
    save_image(blurred_image, os.path.join(args.output_dir, f"{to_blur_image_name}_blurred.png"))

    

         
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
