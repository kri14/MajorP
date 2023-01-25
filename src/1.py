# import the necessary libraries
import io
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
from args import get_parser
import pickle
from model import get_model
from torchvision import transforms
from output_utils import prepare_output
from PIL import Image

#C:/Users/91991/Downloads/

data_dir = 'stream/'
ngrs_vocab = pickle.load(open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb'))
vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))
ingr_vocab_size = len(ngrs_vocab)
instrs_vocab_size = len(vocab)
output_dim = instrs_vocab_size

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_model(data_dir):

    args = get_parser()
    args.maxseqlen = 15
    args.ingrs_only=False
    model = get_model(args,ingr_vocab_size, instrs_vocab_size)
    # Load the pre-trained model parameters
    model_path = os.path.join(data_dir, 'modelbest.ckpt')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    model.ingrs_only = False
    model.recipe_only = False
    return model



def predict(model,image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    greedy = [True, False, False, False]
    beam = [-1, -1, -1, -1]
    temperature = 1.0
    numgens = len(greedy)

    num_valid = 1
    for i in range(numgens):
        with torch.no_grad():
            outputs = model.sample(input_batch, greedy=greedy[i], 
                                       temperature=temperature, beam=beam[i], true_ingrs=None)
                
        ingr_ids = outputs['ingr_ids'].cpu().numpy()
        recipe_ids = outputs['recipe_ids'].cpu().numpy()
                
        outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ngrs_vocab, vocab)

        if valid['is_valid'] or show_anyways:
            
            st.write('RECIPE', num_valid)
            num_valid+=1
            #print ("greedy:", greedy[i], "beam:", beam[i])
    
            BOLD = '\033[1m'
            END = '\033[0m'
            st.write(BOLD + '\nTitle:' + END,outs['title'])

            st.write(BOLD + '\nIngredients:'+ END)
            st.write(print (', '.join(outs['ingrs'])))
            st.write(BOLD + '\nInstructions:'+END)
            st.write('-'+'\n-'.join(outs['recipe']))

            st.write('='*20)

    else:
            pass
            st.write("Not a valid recipe!")
            st.write("Reason: ", valid['reason'])
    


def main():
    st.title('IMAGE TO RECIPE')
    model = load_model(data_dir)
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        predict(model, image)


if __name__ == '__main__':
    main()
    