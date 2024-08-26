import pandas as pd
import re
import spacy
import neuralcoref
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from pyvis.network import Network
import datetime as dt


nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)


def resolveText(text):
    """
    Returns: Clean and Resolved Text (str), Global Entity List (list(str))
    """
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\[\d+\]', ' ', text)
    text = re.sub(r'\([^()]*\)', ' ', text)
    text = re.sub(r'(?<=[.,])(?=[^\s0-9])', r' ', text)

    text = nlp(text)
    text = nlp(text._.coref_resolved)
    ent_list = []
    for t in text.ents:
        ent_list.append(t.text)
    result_ls = []
    
    for token in text:
        if not token.is_stop and not token.is_punct:
            result_ls.append(str(token))
    
    text = ' '.join(result_ls)

    ent_list = list(set(ent_list))

    return text, ent_list


def extractTriplets(text, global_ents_list, prjName):
    sro_triplets = []
    prev_subj = nlp("")[:]
    prev_obj = nlp("")[:]
    sentences = [sent.strip() for sent in text.split('.')]

    for sent in sentences:
        prev_obj_end = 0
        sent = nlp(sent)
        # nlp.add_pipe(nlp.create_pipe('merge_noun_chunks'))
        ents = list(sent.ents)
        global_ents = []

        main_ents = []  # Named Entities recognised by Spacy (Main)
        addn_ents = []  # Additional named entities (Date/Time/etc.)

        for ent in ents:
            if ent.label_ in ("DATE", "TIME", "MONEY", "QUANTITY"):
                addn_ents.append(ent)
            elif ent.label_ in ("CARDINAL", "ORDINAL", "PERCENT"):
                continue     # Ignore cardinal/ordinal numbers and percentages
            elif ent.label_ in ("PERSON", "NORP", "FAC", "ORG", 
                                    "GPE", "LOC", "PRODUCT", "EVENT", 
                                    "WORK_OF_ART", "LAW", "LANGUAGE"):
                main_ents.append(ent)

        for tok in sent:
            if tok.text.lower() in global_ents_list:
                global_ents.append(sent[tok.i:tok.i+1])
        
        noun_chunks = list(sent.noun_chunks)

        verbs = [tok for tok in sent if tok.pos_ == "VERB"]

        for verb in verbs:          
        # Identify Subject
            subj = None
                    # Find leftmost Main Ent to verb
            for ent in main_ents:
                if ent.end > verb.i:
                    break
                elif ent.end > prev_obj_end:
                    subj = ent
                    rel_start = subj.end
            if subj is None:
                # Find leftmost Global Ent to verb
                for ent in global_ents:
                    if ent.end > verb.i:
                        break
                    elif ent.end > prev_obj_end:
                        subj = ent
                        rel_start = subj.end
            if subj is None:
                        # Find leftmost noun chunk to verb
                for noun_chunk in noun_chunks:
                    if noun_chunk.end > verb.i:
                        break
                    elif noun_chunk.end > prev_obj_end:
                        subj = noun_chunk
                        rel_start = subj.end
            if subj is None:
                    # Find leftmost Additional Ent to verb
                for ent in addn_ents:
                    if ent.end > verb.i:
                        break
                    elif ent.end > prev_obj_end:
                        subj = ent
                        rel_start = subj.end


            obj = None
                    # Find rightmost Main Ent to verb
            for ent in main_ents[::-1]:
                if ent.end <= verb.i:
                    break
                else:
                    obj = ent
                    rel_end = obj.start
            if obj is None:
                        # Find rightmost Global Ent to verb
                for ent in global_ents[::-1]:
                    if ent.end <= verb.i:
                        break
                    elif ent.text.lower() != verb.text.lower():  
                                # Additional check for global entity not being verb itself!
                        obj = ent
                        rel_end = obj.start
            if obj is None:
                        # Find rightmost noun chunk to verb
                for noun_chunk in noun_chunks[::-1]:
                    if noun_chunk.end <= verb.i:
                        break
                    else:
                        obj = noun_chunk
                        rel_end = obj.start
            if obj is None:
                        # Find rightmost Additional Ent to verb
                for ent in addn_ents[::-1]:
                    if ent.end <= verb.i:
                        break
                    else:
                        obj = ent
                        rel_end = obj.start
            if obj is None:
                        # If no object found, assign previous subject
                obj = prev_obj
                rel_end = verb.i + 1
            
            if subj is not None:
                triplet = (
                        # Subject
                        " ".join(tok.text.lower() for tok in subj).strip(), 
                        # Relationship
                        " ".join(tok.lemma_.lower() for tok in sent[rel_start:rel_end]).strip(),
                        # Object 
                        " ".join(tok.text.lower() for tok in obj).strip(), 
                    )

            if triplet[0] != "" and triplet[1] != "" and triplet[2] != "" and triplet[0] != triplet[2]:
                        # Check for duplicate triplets within same sentence 
                if subj == prev_subj and obj == prev_obj:
                    prev_triplet = sro_triplets.pop()
                    # Define relation as the longest relation span among duplicates
                    if len(prev_triplet[1]) > len(triplet[1]):
                        triplet = prev_triplet
                        
                sro_triplets.append(triplet)

            prev_subj = subj
            prev_obj = obj
            prev_obj_end = obj.end

    sro_triplets_df = pd.DataFrame(sro_triplets, columns=['subject', 'relation', 'object'])
    sro_triplets_df = sro_triplets_df.drop_duplicates()
    print("\n\n-- Entity Pairs Extracted")
    csv = input("Do you want a csv file?")
    if csv=='y' or csv=='Y':
        dnt = dt.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        sro_triplets_df.to_csv(dnt+" "+prjName + '.csv')
        print("\n\n-- CSV File Generated")
    return sro_triplets_df

def extract_ner_bert(text, model=None, tokenizer=None):
    if model is None:
        model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        
    label_list = [
        "O",       # Outside of a named entity
        "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
        "I-MISC",  # Miscellaneous entity
        "B-PER",   # Beginning of a person's name right after another person's name
        "I-PER",   # Person's name
        "B-ORG",   # Beginning of an organisation right after another organisation
        "I-ORG",   # Organisation
        "B-LOC",   # Beginning of a location right after another location
        "I-LOC"    # Location
    ]
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    inputs = tokenizer.encode(text, return_tensors="pt")

    outputs = model(inputs)[0]
    predictions = torch.argmax(outputs, dim=2)
    
    # Build list of named entities
    ents = []
    cur_ent = ""
    cur_label = ""
    for token, prediction in zip(tokens, predictions[0].tolist()):
        if label_list[prediction] != "O":
            if token[:2] == "##":
                # Append token to current NE
                cur_ent += token[2:]
            else:
                # Start of a new NE
                if cur_ent != "":
                    ents.append(cur_ent)
                cur_ent = token
                cur_label = label_list[prediction]
    if cur_ent != "":
        ents.append(cur_ent)
        
    return ents

def extractTripletsBert(text, global_ents_list):
    sro_triplets = []
    prev_subj = nlp("")[:]
    prev_obj = nlp("")[:]

    sentences = [sent.strip() for sent in text.split('.')]

    for sent in sentences:
        prev_obj_end = 0
        sent = nlp(sent)
        # nlp.add_pipe(nlp.create_pipe('merge_noun_chunks'))
        ents = list(sent.ents)
        spans = spacy.util.filter_spans(ents)
        with sent.retokenize() as retokenizer:
            [retokenizer.merge(span) for span in spans]
            
        ents = extract_ner_bert(sent)
        main_ents = []  # Named Entities recognised by Spacy (Main)
        addn_ents = []  # Additional named entities (Date/Time/etc.)

        for ent in ents:
            if ent.label_ in ("DATE", "TIME", "MONEY", "QUANTITY"):
                addn_ents.append(ent)
            elif ent.label_ in ("CARDINAL", "ORDINAL", "PERCENT"):
                continue     # Ignore cardinal/ordinal numbers and percentages
            elif ent.label_ in ("PERSON", "NORP", "FAC", "ORG", 
                                    "GPE", "LOC", "PRODUCT", "EVENT", 
                                    "WORK_OF_ART", "LAW", "LANGUAGE"):
                main_ents.append(ent)
        
        global_ents = []
        
        for tok in sent:
            if tok.text.lower() in global_ents_list:
                global_ents.append(sent[tok.i:tok.i+1])
        
        noun_chunks = list(sent.noun_chunks)

        verbs = [tok for tok in sent if tok.pos_ == "VERB"]

        for verb in verbs:          
        # Identify Subject
            subj = None
                    # Find leftmost Main Ent to verb
            for ent in main_ents:
                if ent.end > verb.i:
                    break
                elif ent.end > prev_obj_end:
                    subj = ent
                    rel_start = subj.end
            if subj is None:
                # Find leftmost Global Ent to verb
                for ent in global_ents:
                    if ent.end > verb.i:
                        break
                    elif ent.end > prev_obj_end:
                        subj = ent
                        rel_start = subj.end
            if subj is None:
                        # Find leftmost noun chunk to verb
                for noun_chunk in noun_chunks:
                    if noun_chunk.end > verb.i:
                        break
                    elif noun_chunk.end > prev_obj_end:
                        subj = noun_chunk
                        rel_start = subj.end
            if subj is None:
                    # Find leftmost Additional Ent to verb
                for ent in addn_ents:
                    if ent.end > verb.i:
                        break
                    elif ent.end > prev_obj_end:
                        subj = ent
                        rel_start = subj.end

            # if subj is None:
            #     # If no subject found, assign default subject
            #     subj = 'default_subj'
            #     rel_start = verb.i


            obj = None
                    # Find rightmost Main Ent to verb
            for ent in main_ents[::-1]:
                if ent.end <= verb.i:
                    break
                else:
                    obj = ent
                    rel_end = obj.start
            if obj is None:
                        # Find rightmost Global Ent to verb
                for ent in global_ents[::-1]:
                    if ent.end <= verb.i:
                        break
                    elif ent.text.lower() != verb.text.lower():  
                                # Additional check for global entity not being verb itself!
                        obj = ent
                        rel_end = obj.start
            if obj is None:
                        # Find rightmost noun chunk to verb
                for noun_chunk in noun_chunks[::-1]:
                    if noun_chunk.end <= verb.i:
                        break
                    else:
                        obj = noun_chunk
                        rel_end = obj.start
            if obj is None:
                        # Find rightmost Additional Ent to verb
                for ent in addn_ents[::-1]:
                    if ent.end <= verb.i:
                        break
                    else:
                        obj = ent
                        rel_end = obj.start
            if obj is None:
                        # If no object found, assign previous subject
                obj = prev_obj
                rel_end = verb.i + 1
            
            if subj is not None:
                triplet = (
                        # Subject
                        " ".join(tok.text.lower() for tok in subj).strip(), 
                        # Relationship
                        " ".join(tok.lemma_.lower() for tok in sent[rel_start:rel_end]).strip(),
                        # Object 
                        " ".join(tok.text.lower() for tok in obj).strip(), 
                    )

            if triplet[0] != "" and triplet[1] != "" and triplet[2] != "" and triplet[0] != triplet[2]:
                        # Check for duplicate triplets within same sentence 
                if subj == prev_subj and obj == prev_obj:
                    prev_triplet = sro_triplets.pop()
                    # Define relation as the longest relation span among duplicates
                    if len(prev_triplet[1]) > len(triplet[1]):
                        triplet = prev_triplet
                        
                sro_triplets.append(triplet)

            prev_subj = subj
            prev_obj = obj
            prev_obj_end = obj.end

    sro_triplets_df = pd.DataFrame(sro_triplets, columns=['subject', 'relation', 'object'])
    return sro_triplets_df

def merge_duplicate_subjs(triplets):
    subjects = sorted(list(triplets.subject.unique()))
    prev_subj = subjects[0]
    for subj in subjects[1:]:
        # TODO Use string edit distance between prev_subj and subj
        if prev_subj in subj:
            # Detect extension in subj compared to prev_subj and append it to relations of rows with subj
            triplets.loc[triplets.subject==subj, 'relation'] = subj.replace(prev_subj, '').strip() + ' ' + triplets[triplets.subject==subj].relation
            # Update subject from subj to prev_subj
            triplets.loc[triplets.subject==subj, 'subject'] = prev_subj
            
        else:
            # Update prev_subj
            prev_subj = subj
    
    return triplets

def prune_infreq_subjects(triplets, threshold=1):
    triplets = merge_duplicate_subjs(triplets)
    # Count unique subjects
    subj_counts = triplets.subject.value_counts()
    # TODO: add more/smarter heuristics for pruning?
    # Drop subjects with counts below threshold
    triplets['subj_count'] = list(subj_counts[triplets.subject])
    triplets.drop(triplets[triplets['subj_count'] < threshold].index, inplace=True)
    triplets = triplets.drop('subj_count', 1)
    return triplets


def prune_infreq_objects(triplets, threshold=1):
    # Count unique objects
    obj_counts = triplets.object.value_counts()
    # TODO: add more/smarter heuristics for pruning?
    # Drop objects with counts below threshold
    triplets['obj_count'] = list(obj_counts[triplets.object])
    triplets.drop(triplets[triplets['obj_count'] < threshold].index, inplace=True)
    triplets = triplets.drop('obj_count', 1)
    return triplets


def prune_self_loops(triplets):
    """Helper function to prune triplets where subject is the same as object
    """
    triplets.drop(triplets[triplets.subject==triplets.object].index, inplace=True)
    
    return triplets.drop_duplicates()

def generateGraph(triplets, prjName):
    dnt = dt.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    net = Network()
    for i in range(len(triplets)):
        net.add_node(triplets["subject"].iloc[i])
        net.add_node(triplets["object"].iloc[i])
        net.add_edge(triplets["subject"].iloc[i], triplets["object"].iloc[i])
    
    net.write_html(dnt+" "+prjName+'.html', notebook=False, open_browser=False)
    print("\n\nGraph Generated.\n")
    return None
