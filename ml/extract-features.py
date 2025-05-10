#! /usr/bin/python3

import sys
from os import listdir
from typing import Dict, TypedDict
from xml.dom.minidom import parse

from deptree import deptree

# import patterns

OffsetDict = TypedDict(start=int, end=int)

## -------------------
## -- Convert a pair of drugs and their context in a feature vector


def extract_features(dependency_tree: deptree, entities: Dict[str, OffsetDict], entity_1: str, entity_2: str):
    feats: Dict[str, str] = {}

    # get head token for each gold entity
    tkE1 = dependency_tree.get_fragment_head(entities[entity_1]["start"], entities[entity_1]["end"])
    tkE2 = dependency_tree.get_fragment_head(entities[entity_2]["start"], entities[entity_2]["end"])

    if tkE1 is not None and tkE2 is not None:
        # features for tokens in between E1 and E2
        # for tk in range(tkE1+1, tkE2) :
        tk = tkE1 + 1
        try:
            while dependency_tree.is_stopword(tk):
                tk += 1
        except:
            return {}
        word = dependency_tree.get_word(tk)
        lemma = dependency_tree.get_lemma(tk).lower()
        tag = dependency_tree.get_tag(tk)
        feats.add("lib=" + lemma)
        feats.add("wib=" + word)
        feats.add("lpib=" + lemma + "_" + tag)

        eib = False
        for tk in range(tkE1 + 1, tkE2):
            if dependency_tree.is_entity(tk, entities):
                eib = True

        # feature indicating the presence of an entity in between E1 and E2
        feats.add("eib=" + str(eib))

        # features about paths in the tree
        lcs = dependency_tree.get_LCS(tkE1, tkE2)

        path1 = dependency_tree.get_up_path(tkE1, lcs)
        path1 = "<".join([dependency_tree.get_lemma(x) + "_" + dependency_tree.get_rel(x) for x in path1])
        feats.add("path1=" + path1)

        path2 = dependency_tree.get_down_path(lcs, tkE2)
        path2 = ">".join([dependency_tree.get_lemma(x) + "_" + dependency_tree.get_rel(x) for x in path2])
        feats.add("path2=" + path2)

        path = path1 + "<" + dependency_tree.get_lemma(lcs) + "_" + dependency_tree.get_rel(lcs) + ">" + path2
        feats.add("path=" + path)

    return feats


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  extract_features targetdir
## --
## -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
## --

# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir):

    # parse XML file, obtaining a DOM tree
    xml_tree = parse(datadir + "/" + f)

    # process each sentence in the file
    xml_sentences = xml_tree.getElementsByTagName("sentence")
    for xml_sentence in xml_sentences:
        sentence_id = xml_sentence.attributes["id"].value  # get sentence id
        sentence_text = xml_sentence.attributes["text"].value  # get sentence text
        # load sentence entities
        entities: Dict[str, OffsetDict] = {}
        xml_entities = xml_sentence.getElementsByTagName("entity")
        for xml_entity in xml_entities:
            entity_id = xml_entity.attributes["id"].value
            entity_offsets = xml_entity.attributes["charOffset"].value.split("-")
            entities[entity_id] = {"start": int(entity_offsets[0]), "end": int(entity_offsets[-1])}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1:
            continue

        # analyze sentence
        dependency_tree = deptree(sentence_text)

        # for each pair in the sentence, decide whether it is DDI and its type
        xml_pairs = xml_sentence.getElementsByTagName("pair")
        for xml_pair in xml_pairs:
            # ground truth
            is_interaction = xml_pair.attributes["ddi"].value
            if is_interaction == "true":
                interaction_type = xml_pair.attributes["type"].value
            else:
                interaction_type = "null"
            # target entities
            interaction_entity_1 = xml_pair.attributes["e1"].value
            interaction_entity_2 = xml_pair.attributes["e2"].value
            # feature extraction

            features = extract_features(dependency_tree, entities, interaction_entity_1, interaction_entity_2)
            # resulting vector
            if len(features) != 0:
                print(
                    sentence_id,
                    interaction_entity_1,
                    interaction_entity_2,
                    interaction_type,
                    "\t".join(f"{k}={v}" for k, v in features.items()),
                    sep="\t",
                )
