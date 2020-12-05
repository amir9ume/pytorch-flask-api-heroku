import json
import os
import torch

from commons import get_model #, transform_image
model, tokenizer = get_model()

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)

def get_prediction(image_bytes):
    try:
        
        #tensor = transform_image(image_bytes=image_bytes)        
        inputs = tokenizer([image_bytes], return_tensors='pt') 
        #print('inputs are : ',inputs)
        reply_ids = model.generate(inputs['input_ids'])
        #print('reply ids : ',reply_ids)
        replies= ([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in reply_ids])
        
    except Exception:
        return 0, 'error'
    
    return replies
    