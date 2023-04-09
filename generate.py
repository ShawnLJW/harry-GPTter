import argparse
import os
import tiktoken
import torch
from pytorch_lightning import seed_everything
from model import GenerativeTransformer, ModelConfig

def load_model(path, device):
    print(f"Loading model from {path}")
    config = ModelConfig()
    model = GenerativeTransformer(config).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
    
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--prompt',
        type=str,
        default='Two hundred miles away, the boy called Harry Potter',
        help='the generated text will start with the prompt'
    )
    
    parser.add_argument(
        '--outdir',
        type=str,
        default='outputs',
        help='directory output will be written to'
    )
    
    parser.add_argument(
        '--ckpt',
        type=str,
        default='checkpoints/checkpoint.pt',
        help='path to model weights'
    )
    
    parser.add_argument(
        '--length',
        type=int,
        default=500,
        help='how many tokens to generate for each sample'
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=3,
        help='how many samples to produce (batch size)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=72,
        help='seed for reproducible sampling'
    )
    
    opt = parser.parse_args()
    
    enc = tiktoken.get_encoding('p50k_base')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(opt.ckpt, device)
    seed_everything(opt.seed)
    
    os.makedirs(opt.outdir, exist_ok=True)
    directory_path = opt.outdir
    
    prompt = torch.tensor(enc.encode(opt.prompt))
    prompt = prompt.repeat(opt.n_samples, 1).to(device)
    gen_text = model.generate(prompt, opt.length)
    gen_text = [enc.decode(i) for i in gen_text.cpu().numpy()]
    
    base_count = len(os.listdir(directory_path))
    for sample in gen_text:
        base_count += 1
        sample_path = os.path.join(directory_path,f'sample-{base_count}.txt')
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write(sample)
    
if __name__ == '__main__':
    main()
    