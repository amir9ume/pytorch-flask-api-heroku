import io

from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration

def get_model():
    mname = 'facebook/blenderbot-90M'
    model = BlenderbotForConditionalGeneration.from_pretrained(mname)
    print('Loading tokenizer...')
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
    print('init done')
    return model, tokenizer


def format_class_name(class_name):
    class_name= class_name[0]
    new_sentence= '. '.join(list(map(lambda x: x.strip().capitalize(), class_name.split('.'))))
    print('class name here ', new_sentence)
    return new_sentence
