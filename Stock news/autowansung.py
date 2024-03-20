


class TrieNode:
    def __init__(self):
        self.word_num = 0      					# 현재 문자를 포함하는 단어 수
        self.children = defaultdict(TrieNode)   # 자식 노드 (dict 자료형)

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        단어 삽입
        :param word:
        :return:
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node.children[char].word_num += 1   # 해당 문자를 포함하는 단어 수 + 1
            node = node.children[char]          # 다음 노드로 이동


def solution(words):
    answer = 0
    trie = Trie()
    # 1. 트라이에 단어 삽입
    for word in words:
        trie.insert(word)

    # 2. 트라이 구조로 몇 글자를 입력해야하는지 파악
    for word in words:
        cur_node = trie.root
        for i, char in enumerate(word):
            if cur_node.children[char].word_num == 1:  # 자식에 해당 단어가 1인 경우(현재 해당 단어밖에 없는 경우)
                break
            cur_node = cur_node.children[char]   # 다음 단어로 이동
        answer += (i + 1)   # 해당 단어 입력 해야할 수
    return answer
