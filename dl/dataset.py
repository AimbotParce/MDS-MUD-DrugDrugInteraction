import os
import pickle
from typing import Dict, Generator, TypedDict
from xml.dom.minidom import parse

from deptree import *

EntityDict = TypedDict("EntityDict", {"start": int, "end": int, "type": str})
TokenDict = TypedDict(
    "TokenDict",
    {"form": str, "lc_form": str, "lemma": str, "pos": str, "etype": Optional[str]},
)
SentenceDict = TypedDict(
    "SentenceDict",
    {"sid": str, "e1": str, "e2": str, "type": str, "sent": List[TokenDict]},
)


class Dataset:
    """
    Parse all XML files in given dir, and load a list of sentences.
    Each sentence is a list of tuples (word, start, end, tag)
    """

    def __init__(self, filename: str):
        """
        Load the dataset from a directory or a pickle file.

        If a directory is given, it will parse all XML files in the directory.
        If a pickle file is given, it will be assumed to contain a list of
        already parsed sentences.

        Args:
            filename: Either a directory with XML files, or a pickle file.
        """
        if (
            os.path.splitext(filename)[1] == ".pck"
        ):  # If filename is a pickle, it must be a list of sentences already
            with open(filename, "rb") as pf:
                self.data: List[SentenceDict] = pickle.load(pf)
        elif os.path.isdir(
            filename
        ):  # If filename is a directory, it must contain XML files
            self.data: List[SentenceDict] = []
            for f in os.listdir(filename):  # Process each file in directory
                xml_tree = parse(
                    filename + "/" + f
                )  # Parse XML file, obtaining a DOM tree
                xml_sentences = xml_tree.getElementsByTagName(
                    "sentence"
                )  # Process each sentence in the file
                for xml_sentence in xml_sentences:
                    sentence_id = xml_sentence.attributes["id"].value  # get sentence id
                    sentence_text = xml_sentence.attributes[
                        "text"
                    ].value  # get sentence text
                    xml_entities = xml_sentence.getElementsByTagName("entity")

                    if (
                        len(xml_entities) <= 1
                    ):  # If there are no entity pairs, skip sentence
                        continue

                    entities: Dict[str, EntityDict] = {}
                    for xml_entity in xml_entities:
                        # for discontinuous entities, we only get the first span
                        # (will not work, but there are few of them)
                        entity_id = xml_entity.attributes["id"].value
                        entity_type = xml_entity.attributes["type"].value
                        (start, end) = (
                            xml_entity.attributes["charOffset"]
                            .value.split(";")[0]
                            .split("-")
                        )
                        entities[entity_id] = {
                            "start": int(start),
                            "end": int(end),
                            "type": entity_type,
                        }

                    # analyze sentence with stanford parser.
                    dependency_tree = deptree(sentence_text)

                    # for each pair in the sentence, get whether it is DDI and its type
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

                        sentence_tokens: List[TokenDict] = []
                        seen_entities: set[str] = set()
                        for node in range(1, dependency_tree.get_n_nodes()):
                            entity_id = dependency_tree.get_entity_id(node, entities)

                            if entity_id is None:
                                token = {
                                    "form": dependency_tree.get_word(node),
                                    "lc_form": dependency_tree.get_word(node).lower(),
                                    "lemma": dependency_tree.get_lemma(node),
                                    "pos": dependency_tree.get_tag(node),
                                }
                            elif entity_id == interaction_entity_1:
                                token = {
                                    "form": "<DRUG1>",
                                    "lc_form": "<DRUG1>",
                                    "lemma": "<DRUG1>",
                                    "pos": "<DRUG1>",
                                    "etype": entities[interaction_entity_1]["type"],
                                }
                            elif entity_id == interaction_entity_2:
                                token = {
                                    "form": "<DRUG2>",
                                    "lc_form": "<DRUG1>",
                                    "lemma": "<DRUG2>",
                                    "pos": "<DRUG2>",
                                    "etype": entities[interaction_entity_2]["type"],
                                }
                            else:
                                token = {
                                    "form": "<DRUG_OTHER>",
                                    "lc_form": "<DRUG_OTHER>",
                                    "lemma": "<DRUG_OTHER>",
                                    "pos": "<DRUG_OTHER>",
                                    "etype": entities[entity_id]["type"],
                                }

                            if entity_id == None or entity_id not in seen_entities:
                                sentence_tokens.append(token)
                            if entity_id != None:
                                # To avoid duplicates (some entities are split in the middle by the dependency tree
                                # parser, and we would get several tokens for the same entity)
                                seen_entities.add(entity_id)

                        # resulting vector
                        self.data.append(
                            {
                                "sid": sentence_id,
                                "e1": interaction_entity_1,
                                "e2": interaction_entity_2,
                                "type": interaction_type,
                                "sent": sentence_tokens,
                            }
                        )

    def save(self, filename: str):
        """
        Save the dataset to a pickle file.
        """
        with open(filename, "wb") as pf:
            pickle.dump(self.data, pf)

    def sentences(self) -> Generator[SentenceDict, None, None]:
        """
        Iterate over the sentences in the dataset.
        """
        for s in self.data:
            yield s
