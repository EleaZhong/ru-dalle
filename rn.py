import translators
import ruclip
from rudalle.pipelines import generate_images_arb, show, super_resolution, cherry_pick_by_ruclip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything
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

text = 'a full moon at sea, oil on canvas' # @param 

#@markdown *радуга на фоне ночного города / rainbow on the background of the city at night*

if simple_detect_lang(text) != 'ru':
    text = translators.google(text, from_language='en', to_language='ru')
print('text:', text)

import random
seed_everything(random.randint(0,9999))

pil_images = []
ppl_scores = []
for top_k, top_p, images_num, cfg in [
    (2048, 0.995, 1, 4),
]:
    _pil_images, _ = generate_images_arb(text, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, top_p=top_p, bs=8, use_cache=False,true_size=80*80, cfg=cfg)
    pil_images += _pil_images
    # ppl_scores += _ppl_scores
for n,p in enumerate(pil_images):
    p.save(f"{n}.png")