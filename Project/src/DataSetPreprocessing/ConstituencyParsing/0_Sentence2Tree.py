import os
import json
import tqdm
from tags import *
from tree import *
from path import *
from load_file import *
from paths import *

# Sentence to tree dict
def traverse_tree(phrase_tree):
    phrase_dict = {} # dict{text: Str, label: Tuple, child: List}
    phrase_dict['text'] = phrase_tree.text
    phrase_dict['label'] = phrase_tree.label
    
    merging_label = False
    merging_conjuction = True
    while merging_conjuction:
       
        phrase_dict['children'] = []
       # print(phrase_tree.children,"CHILDREN")
        for c_idx, child in enumerate(phrase_tree.children):
           # print(c_idx,child,"OK??")
            if merging_label:
                continue
            
            child_dict = traverse_tree(child)

            if not label_is_pos_punct(child_dict['label']) and \
                not label_is_pos_special(child_dict['label']) and \
                not label_is_benepar_valid(child_dict['label']): # Stop Parsing invalid pos and invalid phrase
                phrase_dict['children'] = []
                break

            elif label_is_conjuction(child_dict['label']) or \
                    is_stopwords_phrase(child_dict['text']): # Merge conjunction, stopwords to right
                
                if c_idx == len(phrase_tree.children) - 1: # rightest
                    if len(phrase_tree.children) == 1 :
                        phrase_dict['children'] = []
                        break
                    elif len(phrase_tree.children) == 2 :
                        if len(phrase_tree.children[c_idx - 1].children) > 0:
                            phrase_tree = upgrade_with_left_tree(phrase_tree, phrase_tree.children[c_idx - 1], phrase_tree.children[c_idx])
                            merging_label = True
                        else:
                            phrase_dict['children'] = []
                            break
                    else: # len(phrase_tree.children) > 2
                        phrase_tree.children[c_idx - 1] = merge_to_left_tree(phrase_tree.children[c_idx - 1], phrase_tree.children[c_idx])
                        _ = phrase_tree.children.pop(c_idx)
                        merging_label = True
                
                else: # some right nodes remains
                    if len(phrase_tree.children) == 2 :
                        if len(phrase_tree.children[c_idx + 1].children) > 0:
                            phrase_tree = upgrade_with_right_tree(phrase_tree, phrase_tree.children[c_idx + 1], phrase_tree.children[c_idx])
                            merging_label = True
                        else:
                            phrase_dict['children'] = []
                            break
                    else: # len(phrase_tree.children) > 2
                        phrase_tree.children[c_idx + 1] = merge_to_right_tree(phrase_tree.children[c_idx + 1], phrase_tree.children[c_idx])
                        _ = phrase_tree.children.pop(c_idx)
                        merging_label = True

            elif label_is_pos_punct(child_dict['label']): # Merge punct to left
                
                if c_idx == 0: # leftest
                    if len(phrase_tree.children) == 2 :
                        if len(phrase_tree.children[c_idx + 1].children) > 0:
                            phrase_tree = upgrade_with_right_tree(phrase_tree, phrase_tree.children[c_idx + 1], phrase_tree.children[c_idx])
                            merging_label = True
                        else:
                            phrase_dict['children'] = []
                            break
                    else: # len(phrase_tree.children) > 2
                       # print(phrase_tree.children,c_idx,"OK INSIDE???")
                        phrase_tree.children[c_idx + 1] = merge_to_right_tree(phrase_tree.children[c_idx + 1], phrase_tree.children[c_idx])
                        _ = phrase_tree.children.pop(c_idx)
                        merging_label = True
                
                else: # some left nodes remains
                    if len(phrase_tree.children) == 2 :
                        if len(phrase_tree.children[c_idx - 1].children) > 0:
                            phrase_tree = upgrade_with_left_tree(phrase_tree, phrase_tree.children[c_idx - 1], phrase_tree.children[c_idx])
                            merging_label = True
                        else:
                            phrase_dict['children'] = []
                            break
                    else: # len(phrase_tree.children) > 2
                        phrase_tree.children[c_idx - 1] = merge_to_left_tree(phrase_tree.children[c_idx - 1], phrase_tree.children[c_idx])
                        _ = phrase_tree.children.pop(c_idx)
                        merging_label = True
            else:
                phrase_dict['children'].append(child_dict)

        if merging_label:
            merging_label = False
        else:
            merging_conjuction = False
    # print(phrase_dict)
        return phrase_dict



def get_tree_from_sentence(nlp, text):
    # input: parser, text, writing path
    # output: 
    text = text.strip()
    # text = text.lower()
    doc = nlp(text)
    sents = list(doc.sents)

    sent_dict_list = []
    for sent in sents:
        string_sentence = str(sent)
        print(string_sentence,"OK STRING")
        if(string_sentence == "."):
           continue
        # print(sent._.parse_string)
        sent_tree = construct_constituency_tree_from_string(sent._.parse_string)
        sent_dict = traverse_tree(sent_tree)
        sent_dict_list.append(sent_dict)
        # print(sent_dict)
    return sent_dict_list

def save_tree_dict(nlp, texts_dict, dst_path,entries_dict):
    for text_id in tqdm.tqdm(texts_dict):
        if(entries_dict.get(text_id,False)):
            continue
        w_folder = os.path.join(dst_path, text_id)
        if not os.path.exists(w_folder):
            os.makedirs(w_folder)
        print(f"Parsing {text_id}")
        index = 0
        for t_id, text in enumerate(texts_dict[text_id]):
            w_file = os.path.join(w_folder, str(t_id).zfill(3) + '.json')
            print(f"Parsing {text_id} {t_id} - {index}")
            dict_list = get_tree_from_sentence(nlp, text)
            write_tree_list_file(w_file, dict_list)

def get_parsed_trees_so_far(path):
    entries = os.listdir(path)
    dictionary = {}
    for entry in entries:
        dictionary[entry] = True
    return dictionary

if __name__ == "__main__":
    import sys


    file_name = TEXT2SHAPE_PATH
    dst_path = PARSED_TREE_DICT_PATH

    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)

    entries_dict = get_parsed_trees_so_far(dst_path)
    # read text2shape csv file
    texts_dict, texts = get_all_valid_data_text2shape(file_name)
    # texts = ['A lounge or sofa chair.It is double cushion chair.It is grey and golden colour.The back rest is very spacious.']
    #print(len(texts),texts_dict,"TEXTs")
    # parse csv file
    parser = get_BerkeleyNeuralParser()
    # text = texts[6296]
    # for i in range(6296,7200):
    #     output = get_tree_from_sentence(parser,texts[i])
    #     print(output,"OUTPUT DONE")
    # print(output,"OUTPUT")
    
    
    # parse_sentence_to_tree(parser, texts[0])
    save_tree_dict(parser, texts_dict, dst_path,entries_dict)