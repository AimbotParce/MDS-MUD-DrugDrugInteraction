import sys
from typing import Dict, List, Optional, Set, Tuple, TypedDict

from nltk.parse.corenlp import CoreNLPDependencyParser

dep_parser = CoreNLPDependencyParser(url="http://localhost:9000")

OffsetDict = TypedDict("OffsetDict", {"start": int, "end": int})


class deptree:

    ## --------------------------------------------------------------
    ## analyze a sentence with stanforCore and get a dependency tree
    def __init__(self, txt: str):
        if txt == "":
            self.tree = None
        else:
            txt2 = txt.replace("/", " / ").replace("-", " - ").replace(".", ". ").replace("'", " ' ")
            self.tree = next(dep_parser.raw_parse(txt2))
            offset = 0
            for t in self.get_nodes():
                # enrich tree nodes with offset in original text.
                word = self.tree.nodes[t]["word"]
                offset = txt.find(word, offset)
                self.tree.nodes[t]["start"] = offset
                self.tree.nodes[t]["end"] = offset + len(word) - 1
                offset += len(word)

    def get_nodes(self) -> List[int]:
        """
        Get the ids of the nodes in the tree, excluding the root node (0).
        """
        return sorted(self.tree.nodes)[1:]

    def get_n_nodes(self) -> int:
        """
        Get the number of nodes in the tree, including the root node (0).
        """
        return len(self.tree.nodes)

    def get_ancestors(self, n: int) -> List[int]:
        """
        Get the list of ancestors of a node, including the node itself.
        """
        anc: List[int] = []
        while n != 0:
            anc.append(n)
            n = self.tree.nodes[n]["head"]
        return anc

    def get_parent(self, n: int) -> int:
        """
        Get the parent of a node.
        """
        if n == 0:
            return None
        else:
            return self.tree.nodes[n]["head"]

    def get_children(self, n: int) -> List[int]:
        """
        Get the children of a node.
        """
        if self.tree is None:
            return []
        return [c for c in self.tree.nodes if self.get_parent(c) == n]

    def get_LCS(self, n1: int, n2: int) -> int:
        """
        Get the Lowest Common Subsumer (LCS) of two nodes.
        This is the first common ancestor of the two nodes in the tree.
        """
        # get ancestor list for each node
        a1 = self.get_ancestors(n1)
        a2 = self.get_ancestors(n2)
        # get first common element in both lists
        for i in range(len(a1)):
            for j in range(len(a2)):
                if a1[i] == a2[j]:
                    return a1[i]

        # (should never happen since tree root is always a common subsumer.)
        return None

    def get_fragment_head(self, start: int, end: int):
        """
        Given a sentence fragment (start, end), get the token heading it.
        The token is the one that is the lowest common subsumer of all tokens
        overlapping the fragment, which must be also overlapping the fragment.
        If no token is found, return None.
        """

        # find which tokens overlap the fragment
        tokens_overlapping: Set[int] = set()
        for t in self.tree.nodes:
            tk_start, tk_end = self.get_offset_span(t)
            if tk_start <= start <= tk_end or tk_start <= end <= tk_end:
                tokens_overlapping.add(t)

        head: Optional[int] = None
        if len(tokens_overlapping) > 0:
            # find head node among those overlapping the entity
            for t in tokens_overlapping:
                if head is None:
                    head = t
                else:
                    head = self.get_LCS(head, t)

            # if found LCS does not overlap the entity, the parsing was wrong, forget it.
            if head not in tokens_overlapping:
                head = None

        return head

    def get_word(self, n: int) -> str:
        """
        Get the word form of a node from the original text.
        """
        return self.tree.nodes[n]["word"] if self.tree.nodes[n]["word"] is not None else "<none>"

    def get_lemma(self, n: int) -> str:
        """
        Get the lemma of a node from the original text, as reported by the CoreNLP parser.
        """
        return self.tree.nodes[n]["lemma"] if self.tree.nodes[n]["lemma"] is not None else "<none>"

    def get_rel(self, n: int) -> str:
        """
        Get the syntactic function of a node from the original text, as reported by the CoreNLP parser.
        """
        return self.tree.nodes[n]["rel"] if self.tree.nodes[n]["rel"] is not None else "<none>"

    def get_tag(self, n: int) -> str:
        """
        Get the Parts of Speech (PoS) tag of a node from the original text, as reported by the CoreNLP parser.
        """
        return self.tree.nodes[n]["tag"] if self.tree.nodes[n]["tag"] is not None else "<none>"

    def get_offset_span(self, n: int) -> Tuple[int, int]:
        """
        Get the start and end offsets of a node in the original text.
        """
        if n == 0:
            return -1, -1
        else:
            return self.tree.nodes[n]["start"], self.tree.nodes[n]["end"]

    def is_stopword(self, n: int) -> bool:
        """
        Check whether a token is a stopword.
        """
        # if it is not a Noun, Verb, adJective, or adveRb, then it is a stopword
        return self.tree.nodes[n]["tag"][0] not in ["N", "V", "J", "R"]

    def is_entity(self, n: int, entities: Dict[str, OffsetDict]) -> bool:
        """
        Check whether a token belongs to one of the given entities.
        """
        for e in entities:
            if entities[e]["start"] <= self.tree.nodes[n]["start"] and self.tree.nodes[n]["end"] <= entities[e]["end"]:
                return True
        return False

    def get_entity_id(self, n: int, entities: Dict[str, OffsetDict]) -> Optional[str]:
        """
        Get the id of the entity to which a token belongs.
        If the token does not belong to any entity, return None.
        """
        for e in entities:
            if entities[e]["start"] <= self.tree.nodes[n]["start"] and self.tree.nodes[n]["end"] <= entities[e]["end"]:
                return e
        return None

    def get_subtree_offset_span(self, n: int) -> Tuple[int, int]:
        """
        Given a node n, get the start and end offsets of the full subtree rooted at n.
        """
        # if the node is a leaf, get its span
        left, right = self.get_offset_span(n)
        # if it is not a leaf, recurse into leftmost/rightmost children
        children = self.get_children(n)
        if children:
            l, r = self.get_subtree_offset_span(children[0])
            left = min(left, l)
            l, r = self.get_subtree_offset_span(children[-1])
            right = max(right, r)
        return left, right

    def get_up_path(self, n1: int, n2: int) -> Optional[List[int]]:
        """
        Given two nodes n1 and n2 such that n2 is an ancestor of n1,
        return the path from n1 to n2 (excluding n2).

        If n2 is not an ancestor of n1, return None.
        """
        path = self.get_ancestors(n1)
        if n2 not in path:  # error, n2 is not ancestor of n1
            return None
        else:
            return path[: path.index(n2)]

    def get_down_path(self, n1: int, n2: int) -> Optional[List[int]]:
        """
        Given two nodes n1 and n2 such that n1 is an ancestor of n2,
        return the path from n1 to n2 (excluding n1).

        If n1 is not an ancestor of n2, return None.
        """
        path = self.get_up_path(n2, n1)
        if path is not None:  # if None, n1 was not ancestor of n2
            path.reverse()
        return path

    def print(self, n: int = 0, depth: int = 0, file=sys.stdout):
        """
        Print the tree in a readable format.
        """
        if n != 0:
            print(depth * "    ", end="", file=file)
            print(self.get_rel(n) + "(" + self.get_lemma(n) + "_" + self.get_tag(n) + ")", file=file)
        for c in self.get_children(n):
            self.print(c, depth + 1, file=file)
