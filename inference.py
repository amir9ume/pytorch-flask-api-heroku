import json

from commons import get_model #, transform_image



def get_prediction(image_bytes):
    try:
        model, tokenizer = get_model()
        #tensor = transform_image(image_bytes=image_bytes)        
        inputs = tokenizer([image_bytes], return_tensors='pt') 
        #print('inputs are : ',inputs)
        reply_ids = model.generate(inputs['input_ids'])
        #print('reply ids : ',reply_ids)
        replies= ([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in reply_ids])
        
    except Exception:
        return 0, 'error'
    
    return replies
    