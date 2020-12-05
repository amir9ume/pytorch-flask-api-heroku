import io


# from PIL import Image
# from torchvision import models
# import torchvision.transforms as transforms

from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration

def get_model():
    mname = 'facebook/blenderbot-90M'
    model = BlenderbotForConditionalGeneration.from_pretrained(mname)
    print('Loading tokenizer...')
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)
    print('init done')
    return model, tokenizer

# def get_model():
#     model = models.densenet121(pretrained=True)
#     model.eval()
#     return model


# def transform_image(image_bytes):
#     my_transforms = transforms.Compose([transforms.Resize(255),
#                                         transforms.CenterCrop(224),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(
#                                             [0.485, 0.456, 0.406],
#                                             [0.229, 0.224, 0.225])])
#     image = Image.open(io.BytesIO(image_bytes))
#     return my_transforms(image).unsqueeze(0)


def format_class_name(class_name):
    class_name= class_name[0]
    new_sentence= '. '.join(list(map(lambda x: x.strip().capitalize(), class_name.split('.'))))
    print('class name here ', new_sentence)
    #class_name = class_name.replace('_', ' ')
    #class_name = class_name.title()
    return new_sentence
