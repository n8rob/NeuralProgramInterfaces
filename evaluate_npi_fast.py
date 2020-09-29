import numpy as np
import torch
import pickle as pkl
import random

from transformers import *
import run_generation as rg
from torch.nn import functional as F
import copy_of_train_npi_forn8_MAY15 as npi
from train_cat_gan_for_GS_MAY22_GPU1 import NPINetwork, GenerationClassifier
from train_cat_gan_for_GS_MAY22_GPU1 import GPT2WithNPI, GPT2LMWithNPI

import pdb

big_text_file = "/sentences/for/input/text"

def generate_text(in_text, lm_model, tokenizer, target_label=[1],
                      num_generation_iters=75, max_seq_len=10, num_samples=1,  
                      temperature=1, top_k=1, top_p=0.0):
    
    print("Generating text ordinarily",flush=True)

    tokens = tokenizer.encode(in_text)
    # process tokens
    tokens = tokens[-max_seq_len:]
    tokens = torch.tensor(tokens, dtype=torch.long) 
    tokens = tokens.unsqueeze(0).repeat(num_samples, 1) 
    tokens = tokens.to(torch.device('cuda:0'))
    lm_model = lm_model.to(torch.device('cuda:0')) # for some reason this wasn't happening??
    lm_model.transformer.output_hidden_states = False

    num_tokens_needed = max_seq_len - tokens.shape[1]

    out_tokens = []
            
    # We loop through a few times now
    for i in range(num_tokens_needed):

        # Now run the model
        hidden_states, presents = lm_model(input_ids=tokens) 

        # Now we add the new token to the list of tokens
        next_token_logits = hidden_states[0,-1,:] # This is a very long vector
        filtered_logits = rg.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=num_samples)
        next_token_list = next_token.tolist()
        out_tokens = out_tokens + next_token_list
        #next_word = tokenizer.decode(next_token_list)
        #out_text = out_text + " " + next_word
                
        # ...update list of tokens

        tokens = torch.cat((tokens,next_token.unsqueeze(0)),dim=1).to(torch.device('cuda:0'))

    for I in range(num_generation_iters):

        print(".",flush=True,end=" ")

        hidden_states, presents = lm_model(input_ids=tokens)

        # Now we add the new token to the list of tokens
        next_token_logits = hidden_states[0,-1,:] # This is a very long vector
        filtered_logits = rg.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=num_samples)
        next_token_list = next_token.tolist()
        out_tokens = out_tokens + next_token_list
        #next_word = tokenizer.decode(next_token_list)
        #out_text = out_text + " " + next_word
                
        # ...update list of tokens

        tokens = torch.cat((tokens[:,1:],next_token.unsqueeze(0)),dim=1).to(torch.device('cuda:0'))
    
    print("", flush=True)

    return tokenizer.decode(out_tokens)



def generate_text_with_NPI(in_text, lm_model, vanilla_lm_model, tokenizer, perturbation_indices, npi_model, 
                      target_label=[1], num_generation_iters=75, num_seq_iters=10, max_seq_len=10, num_samples=1, 
                      temperature=1, top_k=1, top_p=0.0):

    print("Generating text with NPI perturbations",flush=True)

    lm_model.initialize_npi(perturbation_indices)
    #db.set_trace()
    
    tokens = tokenizer.encode(in_text)
    # process tokens
    tokens = tokens[-max_seq_len:]
    tokens = torch.tensor(tokens, dtype=torch.long) 
    tokens = tokens.unsqueeze(0).repeat(num_samples, 1) 
    tokens = tokens.to(torch.device('cuda:0'))
    lm_model = lm_model.to(torch.device('cuda:0')) # for some reason this wasn't happening??

    vanilla_lm_model.transformer.output_hidden_states = False

    num_tokens_needed = max_seq_len - tokens.shape[1]

    out_tokens = []
            
    # We loop through a few times now
    for i in range(num_tokens_needed):

        # Now run the model
        hidden_states, presents = vanilla_lm_model(input_ids=tokens) 

        # Now we add the new token to the list of tokens
        next_token_logits = hidden_states[0,-1,:] / temperature # This is a very long vector
        filtered_logits = rg.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=num_samples)
        next_token_list = next_token.tolist()
        out_tokens = out_tokens + next_token_list
        #next_word = tokenizer.decode(next_token_list)
        #out_text = out_text + " " + next_word
                
        # ...update list of tokens

        tokens = torch.cat((tokens,next_token.unsqueeze(0)),dim=1).to(torch.device('cuda:0'))

    vanilla_lm_model.transformer.output_hidden_states = True

    while len(out_tokens) < num_generation_iters:

        print(".",flush=True,end=' ')

        big_array = []

        for i in range(num_seq_iters):
            hidden_states, presents, all_hiddens = vanilla_lm_model(input_ids=tokens[:,-max_seq_len:])

            for pi in perturbation_indices:
                big_array.append(all_hiddens[pi])

            next_token_logits = hidden_states[0,-1,:] / temperature
            filtered_logits = rg.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=num_samples)

            tokens = torch.cat((tokens,next_token.unsqueeze(0)),dim=1).to(torch.device('cuda:0'))
        
        tokens = tokens[:,:-num_seq_iters]

        big_array = torch.cat(big_array, dim=1).unsqueeze(3)
        npi_perturbations = npi_model(big_array)
        reshaped = npi_perturbations[:,:,:,0]
        chunked = torch.chunk(reshaped, max_seq_len*len(perturbation_indices), dim=1)
        #db.set_trace()
        curr_perturbs = [x.view(1, max_seq_len, -1) for x in chunked]

        for i in range(num_seq_iters):
            ith_perturbs = curr_perturbs[i*len(perturbation_indices):(i+1)*len(perturbation_indices)]
            # Now run the model
            hidden_states, presents, all_hiddens = \
                lm_model(input_ids=tokens[:,-max_seq_len:], activation_perturbations=ith_perturbs)                         

            # Now we extract the new token and add it to the list of tokens
            next_token_logits = hidden_states[0,-1,:] / temperature
            filtered_logits = rg.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=num_samples)
            next_token_list = next_token.tolist()
            out_tokens = out_tokens + next_token_list
            #next_word = tokenizer.decode(next_token_list)
            #sent = sent + " " + next_word # we just update this so sent remains accurate for dict
            #generated_sent = generated_sent + next_word + " "

            # ...update list of tokens
            tokens = torch.cat((tokens[:,1:],next_token.unsqueeze(0)),dim=1).to(torch.device('cuda:0'))#to(torch.device('cuda:0'))

        tokens = tokens[:,-max_seq_len:]

    print("",flush=True)

    return tokenizer.decode(out_tokens)


if __name__ == "__main__":

    target_word = "cat"

    NPIs_to_test = [
            "/path/to/npi1/",
            "/path/to/npi2/"
            ]

    pis_list = [
            [5,11]
            ] * len(NPIs_to_test)

    
    for ind, (path_to_npi, perturbation_indices) in enumerate(zip(NPIs_to_test, pis_list)):

        print("")
        print("##########################################################")
        print("#### About to start testing for {} with perterub indices {}, test nubmer {} #####".format(path_to_npi, perturbation_indices, ind))
        print("#########################################################")
        print("")

        user_input = ""#input("Press ENTER to proceed or type 'stop' to quit: ")
        if 'stop' in user_input.lower():
            raise KeyboardInterrupt("System quit by user")

        npi_model = torch.load(path_to_npi)

        vanilla_lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
        npi_lm_model = GPT2LMWithNPI.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
        # Make sure everything is on the same GPU
        npi_model = npi_model.to(torch.device('cuda:0'))
        vanilla_lm_model = vanilla_lm_model.to(torch.device('cuda:0'))
        npi_lm_model = npi_lm_model.to(torch.device('cuda:0'))

        #in_texts_list = ["We're not going to be able to do that",
        #                "How",
        #                "Hello how are you",
        #                "The first type",
        #                "I like fish",
        #                "Cats appeared in the alley",
        #                "The supernova eclipsed"
        #                ]

        in_texts_list = []

        big_text_pkl = "/optionally/use/pickle/for/input/sentences/"

        with open(big_text_file,'r') as f:
        #    in_texts_list = pkl.load(f) # UNCOMMENT IF PICKLE USED
        #in_texts_list = [s for s in in_texts_list if len(s) > 3]
        #random.shuffle(in_texts_list)
        #in_texts_list = in_texts_list[:50]
            iterator = 0
            for line in f:
                if len(line) < 3:# or type(line) != str:
                    #db.set_trace()
                    continue
                in_texts_list.append(line)
                iterator += 1
                if iterator > 50:
                    break
    
        total_vanilla_count = 0
        total_perturbed_count = 0
        total_vanilla_word_instances = 0
        total_perturbed_word_instances = 0

        for in_text in in_texts_list:

            vanilla_text = generate_text(in_text, vanilla_lm_model, tokenizer)
            perturbed_text = generate_text_with_NPI(in_text, npi_lm_model, vanilla_lm_model, tokenizer, perturbation_indices, npi_model)

            print("******=========********")
            print("Input text",in_text)

            print("========")
            print("Vanilla_text:", vanilla_text)
            print("========")
            print("Perturbed text:", perturbed_text)
            print("========")

            vanilla_count = vanilla_text.lower().count(target_word)
            perturbed_count = perturbed_text.lower().count(target_word)
            total_vanilla_count += vanilla_count
            total_perturbed_count += perturbed_count
            total_vanilla_word_instances += int(vanilla_count > 0)
            total_perturbed_word_instances += int(perturbed_count > 0)

            print("Instances of {} with untouched GPT-2 output: {}".format(target_word, vanilla_count))
            print("Instances of {} with NPI perturbation: {}".format(target_word, perturbed_count))

        print("============")
        print("TOTAL Instances of {} with untouched GPT-2 output: {}".format(target_word, total_vanilla_count))
        print("TOTAL Instances of {} with NPI perturbation: {}".format(target_word, total_perturbed_count))
        print("TOTAL Ouputs containing {} with untouched GPT-2 output: {}".format(target_word, total_vanilla_word_instances))
        print("TOTAL Outputs containing {} with NPI perturbation: {}".format(target_word, total_perturbed_word_instances))
        print("")

