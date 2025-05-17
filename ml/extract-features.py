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


def extract_features(dependency_tree: deptree, entities: Dict[str, OffsetDict], entity_1: str, entity_2: str, sentence_text: str):
    feats: Dict[str, str] = {}

    node_1 = dependency_tree.get_fragment_head(entities[entity_1]["start"], entities[entity_1]["end"])
    node_2 = dependency_tree.get_fragment_head(entities[entity_2]["start"], entities[entity_2]["end"])

    if node_1 is not None and node_2 is not None:
        # Entity-Level Features
        feats["entity1-lemma"] = dependency_tree.get_lemma(node_1).lower()
        feats["entity2-lemma"] = dependency_tree.get_lemma(node_2).lower()
        feats["entity1-pos"] = dependency_tree.get_tag(node_1)
        feats["entity2-pos"] = dependency_tree.get_tag(node_2)
        feats["same-entity-lemma"] = str(feats["entity1-lemma"] == feats["entity2-lemma"])
        feats["entity-order"] = "before" if node_1 < node_2 else "after" if node_1 > node_2 else "same"

        # Between-token Features
        in_between_tokens = range(min(node_1, node_2) + 1, max(node_1, node_2))
        feats["num-tokens-between"] = str(len(in_between_tokens))
        feats["has-in-between-entity"] = False

        for j in in_between_tokens:
            if dependency_tree.is_entity(j, entities):
                feats["has-in-between-entity"] = True
            elif not dependency_tree.is_stopword(j) and "in-between-lemma" not in feats:
                feats["in-between-lemma"] = dependency_tree.get_lemma(j).lower()
                feats["in-between-word"] = dependency_tree.get_word(j)
                feats["in-between-lemma:pos"] = f"{feats['in-between-lemma']}_{dependency_tree.get_tag(j)}"

        if "in-between-lemma" not in feats:
            logger.warning(f"Entity {entity_1} and {entity_2} have no non-stopword between them")

        # Dependency Path Features
        lcs = dependency_tree.get_LCS(node_1, node_2)
        path_1 = dependency_tree.get_up_path(node_1, lcs)
        path_2 = dependency_tree.get_down_path(lcs, node_2)

        feats["path-length"] = str(len(path_1) + len(path_2))
        feats["lcs-lemma"] = dependency_tree.get_lemma(lcs).lower()
        feats["lcs-pos"] = dependency_tree.get_tag(lcs)

        feats["upward-path-1"] = "<".join(
            f"{dependency_tree.get_lemma(x)}_{dependency_tree.get_rel(x)}" for x in path_1
        )
        feats["upward-path-2"] = ">".join(
            f"{dependency_tree.get_lemma(x)}_{dependency_tree.get_rel(x)}" for x in path_2
        )

        feats["shortest-path"] = (
            feats["upward-path-1"]
            + "<"
            + feats["lcs-lemma"]
            + "_"
            + dependency_tree.get_rel(lcs)
            + ">"
            + feats["upward-path-2"]
        )

        feats["path-edge-sequence"] = "<".join(
            dependency_tree.get_rel(x) for x in path_1
        ) + f"<{dependency_tree.get_rel(lcs)}>" + ">".join(
            dependency_tree.get_rel(x) for x in path_2
        )

        # Surface/Binary Clues
        surface = sentence_text.lower()
        feats["contains-and"] = str(" and " in surface)
        feats["contains-or"] = str(" or " in surface)
        feats["contains-comma"] = str("," in surface)
        feats["ends-in-punctuation"] = str(surface.strip().endswith(("!", "?")))

        # Clue Verb Detection (categorized)
        advising_verbs = {
            "advise", "recommend", "suggest", "caution", "warn", "monitor", "consider",
            "encourage", "instruct", "inform", "alert", "notify", "guide", "urge",
            "counsel", "consult", "propose", "endorse", "prescribe", "indicate",
            "highlight", "stress", "remind", "emphasize", "note", "observe", "support",
            "prompt", "mention", "advocate", "approve", "favor", "prefer", "acknowledge",
            "report", "assure", "reinforce"
        }
        contraindicating_verbs = {
            "contraindicate", "avoid", "prohibit", "prevent", "restrict", "stop", "discontinue",
            "cease", "withdraw", "block", "ban", "forbid", "exclude", "limit", "reduce",
            "discourage", "impede", "counterindicate", "hinder", "proscribe", "halt",
            "curtail", "suspend", "abolish", "terminate", "eliminate", "deny", "suppress",
            "negate", "refuse"
        }
        interaction_verbs = {
            "interact", "affect", "increase", "decrease", "coadminister", "combine",
            "alter", "modulate", "enhance", "reduce", "potentiate", "inhibit", "induce",
            "block", "compete", "mediate", "amplify", "attenuate", "counteract", "impair",
            "strengthen", "synergize", "synergise", "exacerbate", "diminish", "negate",
            "modify", "change", "influence", "impact", "regulate", "facilitate",
            "augment", "offset", "antagonize", "antagonise", "boost", "interfere",
            "mix", "blend"
}

        feats["has-advising-verb"] = False
        feats["has-contraindicating-verb"] = False
        feats["has-generic-interaction-verb"] = False

        for j in in_between_tokens:
            lemma = dependency_tree.get_lemma(j).lower()

            if lemma in advising_verbs:
                feats["advising-verb-lemma"] = lemma
                feats["advising-verb-rel"] = dependency_tree.get_rel(j)
                feats["has-advising-verb"] = True

            if lemma in contraindicating_verbs:
                feats["contraindicating-verb-lemma"] = lemma
                feats["contraindicating-verb-rel"] = dependency_tree.get_rel(j)
                feats["has-contraindicating-verb"] = True

            if lemma in interaction_verbs:
                feats["interaction-verb-lemma"] = lemma
                feats["interaction-verb-rel"] = dependency_tree.get_rel(j)
                feats["has-generic-interaction-verb"] = True
                
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

            features = extract_features(dependency_tree, entities, interaction_entity_1, interaction_entity_2, sentence_text)
            # resulting vector
            print(
                sentence_id,
                interaction_entity_1,
                interaction_entity_2,
                interaction_type,
                json.dumps(features),
                sep="\t",
            )
