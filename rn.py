import torch
import numpy as np
import translators
import ruclip
from rudalle.pipelines import generate_images_arb, show, super_resolution, cherry_pick_by_ruclip, generate_images
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything
from PIL import Image
device = 'cuda'
from rudalle.dalle import MODELS
MODELS.update({
    'Surrealist_XL': dict(
        hf_version='v3',
        description='Surrealist is 1.3 billion params model from the family GPT3-like, '
                    'that was trained on surrealism and Russian.',
        model_params=dict(
            num_layers=24,
            hidden_size=2048,
            num_attention_heads=16,
            embedding_dropout_prob=0.1,
            output_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            image_tokens_per_dim=32,
            text_seq_length=128,
            cogview_sandwich_layernorm=True,
            cogview_pb_relax=True,
            vocab_size=16384 + 128,
            image_vocab_size=8192,
        ),
        repo_id='shonenkov-AI/rudalle-xl-surrealist',
        filename='pytorch_model.bin',
        authors='shonenkovAI',
        full_description='',
    )
})



class ArbImagePrompts:
    '''
    custom mask inputs arb size
    '''

    def __init__(self, pil_image, mask, vae, w, h, device):
        
        self.device = device
        img = self.preprocess_img(pil_image, w, h)
        mask = self.preprocess_mask(mask, w,h)
        self.image_prompts_idx, self.image_tokens = self.get_image_prompts(img, mask, vae)
    

    def preprocess_img(self, pil_img, w, h):
        img = torch.tensor(np.array(pil_img.resize((w*8,h*8),1).convert('RGB')).transpose(2, 0, 1)) / 255.
        img = img.unsqueeze(0).to(self.device, dtype=torch.float32)
        img = (2 * img) - 1
        return img
    

    def preprocess_mask(self, pil_img, w, h):
        # assert pil_img.size == (w,h), f"mask shape mismatch: mask: {pil_img.size}, with w={w} h={h}"
        mask = torch.round(torch.tensor(np.array(pil_img.resize((w,h),0)), dtype=torch.float32))
        mask = mask.unsqueeze(0).to(self.device, dtype=torch.float32)[:,:,:,1]
        return mask
    

    def get_image_prompts(self, img, mask, vae):
        """
        :param img:
        :param mask: is 32x32 tensor with 0's and 1's where 1 corresponds to masked area
        :param vae:
        :return:
        """
        _, _, [_, _, vqg_img] = vae.model.encode(img)

        bs, vqg_img_h, vqg_img_w = vqg_img.shape
        print(bs, vqg_img_h, vqg_img_w)

        image_prompts = vqg_img.reshape((bs, -1))
        image_prompts_idx = np.arange(vqg_img_w * vqg_img_h)
        # print(image_prompts_idx)
        # print(mask.reshape(-1).bool().cpu())
        # assert False,"help"
        image_prompts_idx = set(image_prompts_idx[~mask.reshape(-1).bool().cpu()])
        self.mask = mask.reshape(-1).bool().cpu()
        # import pdb; pdb.set_trace()
        return image_prompts_idx, image_prompts



class MaskedImagePrompts:
    '''
    A custom implementation of RuDALLE's ImagePrompts
    class that allows for custom 32x32 mask inputs
    '''

    def __init__(self, pil_image, mask, vae, device):
        self.device = device
        img = self.preprocess_img(pil_image)
        mask = self.preprocess_mask(mask)
        self.image_prompts_idx, self.image_prompts, self.mask = self.get_image_prompts(img, mask, vae)
        # self.image_tokens = self.image_prompts

    def preprocess_img(self, pil_img):
        img = torch.tensor(np.array(pil_img.resize((256,256),1).convert('RGB')).transpose(2, 0, 1)) / 255.
        img = img.unsqueeze(0).to(self.device, dtype=torch.float32)
        img = (2 * img) - 1
        return img

    def preprocess_mask(self, pil_img):
        mask = torch.round(torch.tensor(np.array(pil_img), dtype=torch.float32))
        mask = mask.unsqueeze(0).to(self.device, dtype=torch.float32)[:,:,:,1]
        return mask

    @staticmethod
    def check_inner_patch(latent_mask: torch.Tensor, x, y):
        right_line = latent_mask[x:, y]
        left_line = latent_mask[:x, y]
        upper_line = latent_mask[x, :y]
        lower_line = latent_mask[x, y:]
        return any(right_line) and any(left_line) and any(upper_line) and any(lower_line)

    def get_image_prompts(self, img, mask, vae):
        """
        :param img:
        :param mask: is 32x32 tensor with 0's and 1's where 1 corresponds to masked area
        :param vae:
        :return:
        """
        _, _, [_, _, vqg_img] = vae.model.encode(img)

        bs, vqg_img_h, vqg_img_w = vqg_img.shape
        print(bs, vqg_img_h, vqg_img_w)

        image_prompts = vqg_img.reshape((bs, -1))
        image_prompts_idx = np.arange(vqg_img_w * vqg_img_h)
        print(image_prompts_idx)
        print(mask.reshape(-1).bool().cpu())
        # assert False,"help"
        image_prompts_idx = set(image_prompts_idx[~mask.reshape(-1).bool().cpu()])
        # import pdb; pdb.set_trace()
        return image_prompts_idx, image_prompts, None

# MaskedImagePrompts(Image.open("ld_inpaint (9).png"), Image.open("ld_mask_inv (5).png"), get_vae(dwt=False,cache_dir="./models").to(device), device=device)
    
# ArbImagePrompts(Image.open("ld_inpaint (9).png"), Image.open("ld_mask_inv (5).png"), get_vae(dwt=False,cache_dir="./models").to(device), w=32, h=32, device=device)
    
# dalle = get_rudalle_model("Malevich", fp16=True, device=device, cache_dir="./models", pretrained=True)
dalle = get_rudalle_model('Surrealist_XL', pretrained=True, fp16=True, device=device, cache_dir="./models")
realesrgan = get_realesrgan('x2', device=device, cache_dir="./models") # x2/x4/x8
tokenizer = get_tokenizer(cache_dir="./models")
vae = get_vae(dwt=False,cache_dir="./models").to(device)  # for stable generations you should use dwt=False
clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device,cache_dir="./models")
clip_predictor = ruclip.Predictor(clip, processor, device, bs=8)
def simple_detect_lang(text):
    if len(set('абвгдежзийклмнопрстуфхцчшщъыьэюяё').intersection(text.lower())) > 0:
        return 'ru'
    if len(set('abcdefghijklmnopqrstuvwxyz').intersection(text.lower())) > 0:
        return 'en'
    return 'other'
#@markdown do not use the English text for inference, you should use translation to Russian, or you can use directly Russian text 

text = 'a path up the floral mountain valley, oil on canvas' # @param 

#@markdown *радуга на фоне ночного города / rainbow on the background of the city at night*

if simple_detect_lang(text) != 'ru':
    text = translators.google(text, from_language='en', to_language='ru')
print('text:', text)

import random
# seed_everything(random.randint(0,9999))



# image_prompts = ArbImagePrompts(Image.open("ld_inpaint (10).png"), Image.open("ld_mask_inv (7).png"), vae, w=64, h=64, device=device)
# arb_image_prompts = ArbImagePrompts(Image.open("ld_inpaint (5).png"), Image.open("ld_mask_inv (4).png"), vae, w=32, h=32, device=device)
arb_image_prompts = ArbImagePrompts(Image.open("ld_inpaint (5).png"), Image.open("ld_mask (6).png"), vae, w=32, h=32, device=device)
# arb_image_prompts = ArbImagePrompts(Image.open("mask32.png"), Image.open("ld_mask_inv (7).png"), vae, w=32, h=32, device=device)
# mask_image_prompts = MaskedImagePrompts(Image.open("ld_inpaint (10).png"), Image.open("ld_mask_inv (7).png"), vae, device=device)

pil_images = []
ppl_scores = []
# try:
for text, top_k, top_p, images_num, cfg in [
    ("flowers",2048, 0.995, 3, 4),
    # ("a portrait of a cute girl, oil painting",2048, 0.995, 1, 3),
]:
    text = translators.google(text, from_language='en', to_language='ru')
    seed_everything(299)
    
    # _pil_images, _ = generate_images_arb(text, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, image_prompts=mask_image_prompts, top_p=top_p, bs=8, use_cache=False,true_size=64*64, cfg=cfg, autoregressive_samp=True)
    # pil_images += _pil_images
    
    _pil_images, _ = generate_images_arb(text, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, image_prompts=arb_image_prompts, top_p=top_p, bs=8, use_cache=False,true_size=32*32, cfg=cfg, autoregressive_samp=False)
    pil_images += _pil_images
    
#     _pil_images, _ = generate_images_arb(text, tokenizer, dalle, vae, top_k=top_k, images_num=1, image_prompts=None, top_p=top_p, bs=8, use_cache=False,true_size=32*32, cfg=cfg, autoregressive_samp=True)
#     pil_images += _pil_images
    
    # _pil_images, _ = generate_images(text, tokenizer, dalle, vae, top_k=top_k, images_num=images_num,image_prompts=mask_image_prompts, top_p=top_p, bs=8)

        # ppl_scores += _ppl_scores
# except:
#     print("skipped")


for n,p in enumerate(pil_images):
    p.save(f"{n}.png")