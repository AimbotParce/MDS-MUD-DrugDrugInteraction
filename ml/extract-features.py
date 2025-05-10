#! /usr/bin/python3

import json
import logging
import sys
from os import listdir
from typing import Dict, TypedDict
from xml.dom.minidom import parse

from deptree import deptree

logging.basicConfig(
    format="(%(asctime)s - %(name)s) %(levelname)s # %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        # logging.StreamHandler(sys.stderr),
        logging.FileHandler("extract-features.log"),
    ],
)
logger = logging.getLogger("extract-features")

OffsetDict = TypedDict("OffsetDict", {"start": int, "end": int})

## -------------------
## -- Convert a pair of drugs and their context in a feature vector


def extract_features(dependency_tree: deptree, entities: Dict[str, OffsetDict], entity_1: str, entity_2: str):
    feats: Dict[str, str] = {}

    # get head token for each gold entity
    node_1 = dependency_tree.get_fragment_head(entities[entity_1]["start"], entities[entity_1]["end"])
    node_2 = dependency_tree.get_fragment_head(entities[entity_2]["start"], entities[entity_2]["end"])

    if node_1 is not None and node_2 is not None:
        feats["entity-1"] = dependency_tree.get_word(node_1).lower()
        feats["entity-2"] = dependency_tree.get_word(node_2).lower()
        feats["entity-1-lemma"] = dependency_tree.get_lemma(node_1).lower()
        feats["entity-2-lemma"] = dependency_tree.get_lemma(node_2).lower()

        feats["has-more-entities-left"] = False
        feats["has-more-entities-between"] = False
        feats["has-more-entities-right"] = False
        feats["has-more-entities"] = False
        if len(entities) > 2:
            feats["has-more-entities"] = True
            for j in dependency_tree.get_nodes():
                if j != node_1 and j != node_2 and dependency_tree.is_entity(j, entities):
                    if j < node_1:
                        feats["has-more-entities-left"] = True
                    elif j > node_2:
                        feats["has-more-entities-right"] = True
                    else:
                        feats["has-more-entities-between"] = True

        # features about paths in the tree
        lcs = dependency_tree.get_LCS(node_1, node_2)
        feats["lowest-common-ancestor-lemma"] = dependency_tree.get_lemma(lcs)
        feats["lowest-common-ancestor-pos-tag"] = dependency_tree.get_tag(lcs)

        upward_path_1 = dependency_tree.get_up_path(node_1, lcs)
        upward_path_2 = dependency_tree.get_up_path(node_2, lcs)

        feats["depth-difference"] = abs(len(upward_path_1) - len(upward_path_2))
        feats["shortest-path-length"] = len(upward_path_1) + len(upward_path_2) + 1

        for i, node in enumerate(upward_path_1):
            if dependency_tree.is_stopword(node):
                continue
            feats[f"upward-path-1-lemma-{i}"] = dependency_tree.get_lemma(node)
            feats[f"upward-path-1-word-{i}"] = dependency_tree.get_word(node).lower()
            feats[f"upward-path-1-pos-{i}"] = dependency_tree.get_tag(node)
            feats[f"upward-path-1-rel-{i}"] = dependency_tree.get_rel(node)

        for i, node in enumerate(upward_path_2):
            if dependency_tree.is_stopword(node):
                continue
            feats[f"upward-path-2-lemma-{i}"] = dependency_tree.get_lemma(node)
            feats[f"upward-path-2-word-{i}"] = dependency_tree.get_word(node).lower()
            feats[f"upward-path-2-pos-{i}"] = dependency_tree.get_tag(node)
            feats[f"upward-path-2-rel-{i}"] = dependency_tree.get_rel(node)

    elif node_1 is None and node_2 is None:
        logger.warning(f"Didn't find a head token for {entity_1} and {entity_2}")
    elif node_1 is None:
        logger.warning(f"Didn't find a head token for {entity_1}")
    elif node_2 is None:
        logger.warning(f"Didn't find a head token for {entity_2}")

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
            print(
                sentence_id,
                interaction_entity_1,
                interaction_entity_2,
                interaction_type,
                json.dumps(features),
                sep="\t",
            )
