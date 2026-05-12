import re
import json
import unicodedata
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=1500):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.inv_vocab = {}
        self.PAD = 0
        self.UNK = 1
        self.BOS = 2
        self.EOS = 3
        self.SEP = 4  # Separator giữa Q và A

    def _get_pairs(self, word):
        pairs = set()
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def fit(self, texts):
        word_freq = Counter()
        for text in texts:
            # Đảm bảo text là Unicode string đúng chuẩn
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            text = unicodedata.normalize('NFC', text.lower().strip())
            for word in text.split():
                # list(word) trên Python 3 str luôn tách đúng ký tự Unicode
                word_freq[' '.join(list(word)) + ' </w>'] += 1

        vocab = dict(word_freq)

        num_merges = self.vocab_size - 256
        for i in range(num_merges):
            pairs = Counter()
            for word, freq in vocab.items():
                word_pairs = self._get_pairs(tuple(word.split()))
                for pair in word_pairs:
                    pairs[pair] += freq
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            self.merges[best] = i

            new_vocab = {}
            bigram = ' '.join(best)
            replacement = ''.join(best)
            for word in vocab:
                new_word = word.replace(bigram, replacement)
                new_vocab[new_word] = vocab[word]
            vocab = new_vocab

        all_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
            '<SEP>': 4,
        }
        for word in vocab:
            for token in word.split():
                if token not in all_tokens:
                    all_tokens[token] = len(all_tokens)
        self.vocab = all_tokens
        # inv_vocab: key phải là int
        self.inv_vocab = {v: k for k, v in all_tokens.items()}
        print(f"Vocab size: {len(self.vocab)}")

    def _tokenize_word(self, word):
        word_chars = list(word) + ['</w>']
        while True:
            pairs = self._get_pairs(word_chars)
            if not pairs:
                break
            best = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
            if best not in self.merges:
                break
            i = 0
            new_chars = []
            while i < len(word_chars):
                if (i < len(word_chars) - 1 and
                    word_chars[i] == best[0] and
                    word_chars[i+1] == best[1]):
                    new_chars.append(best[0] + best[1])
                    i += 2
                else:
                    new_chars.append(word_chars[i])
                    i += 1
            word_chars = new_chars
        return word_chars

    def encode(self, text, add_special=True):
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        text = unicodedata.normalize('NFC', text.lower().strip())
        ids = [self.BOS] if add_special else []
        for word in text.split():
            tokens = self._tokenize_word(word)
            for token in tokens:
                ids.append(self.vocab.get(token, self.UNK))
        if add_special:
            ids.append(self.EOS)
        return ids

    def encodeQ(self, question):
        """Encode câu hỏi với BOS và SEP ở cuối (dùng khi generate)"""
        if isinstance(question, bytes):
            question = question.decode('utf-8')
        question = unicodedata.normalize('NFC', question.lower().strip())
        ids = [self.BOS]
        for word in question.split():
            tokens = self._tokenize_word(word)
            for token in tokens:
                ids.append(self.vocab.get(token, self.UNK))
        ids.append(self.SEP)
        return ids

    def encode_qa(self, question, answer):
        """Encode cặp Q&A với separator rõ ràng"""
        if isinstance(question, bytes):
            question = question.decode('utf-8')
        if isinstance(answer, bytes):
            answer = answer.decode('utf-8')
        q_ids = self.encode(question, add_special=False)
        a_ids = self.encode(answer, add_special=False)
        # Format: BOS + Q tokens + SEP + A tokens + EOS
        return [self.BOS] + q_ids + [self.SEP] + a_ids + [self.EOS]

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i in (self.BOS, self.EOS, self.PAD):
                continue
            if i == self.SEP:
                tokens.append(' | ')
                continue
            tokens.append(self.inv_vocab.get(i, '<UNK>'))
        text = ''.join(tokens).replace('</w>', ' ').strip()
        return text

    def decode_answer(self, ids):
        """
        Decode danh sách token ids thành text.
        Nếu ids còn chứa SEP (tức là toàn bộ chuỗi BOS+Q+SEP+A+EOS),
        thì chỉ lấy phần sau SEP. Nếu không có SEP thì decode toàn bộ.
        """
        try:
            sep_pos = ids.index(self.SEP)
            answer_ids = ids[sep_pos + 1:]
        except ValueError:
            # Không tìm thấy SEP → ids đã là phần answer rồi
            answer_ids = ids
        return self.decode(answer_ids)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'merges': {str(k): v for k, v in self.merges.items()},
                'special_tokens': {
                    'PAD': self.PAD,
                    'UNK': self.UNK,
                    'BOS': self.BOS,
                    'EOS': self.EOS,
                    'SEP': self.SEP,
                }
            }, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer đã lưu vào {path}")

    def load(self, path):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        # inv_vocab: key phải là int (JSON trả về str cho key dict)
        self.inv_vocab = {int(v): k for k, v in self.vocab.items()}
        self.merges = {eval(k): v for k, v in data['merges'].items()}
        if 'special_tokens' in data:
            sp = data['special_tokens']
            self.PAD = sp.get('PAD', 0)
            self.UNK = sp.get('UNK', 1)
            self.BOS = sp.get('BOS', 2)
            self.EOS = sp.get('EOS', 3)
            self.SEP = sp.get('SEP', 4)


if __name__ == '__main__':
    # Mở file với encoding UTF-8 tường minh
    with open('data/dataset.txt', encoding='utf-8') as f:
        texts = f.readlines()

    tok = BPETokenizer(vocab_size=2000)
    tok.fit(texts)
    tok.save('data/tokenizer.json')

    # Kiểm tra
    test_q = "xin chào"
    test_a = "chào bạn ơi"
    ids = tok.encode_qa(test_q, test_a)
    print(f"Q: {test_q}")
    print(f"A: {test_a}")
    print(f"Encoded: {ids}")
    print(f"Decoded full: {tok.decode(ids)}")
    print(f"Decoded answer: {tok.decode_answer(ids)}")

    # Test encodeQ
    q_ids = tok.encodeQ(test_q)
    print(f"encodeQ: {q_ids}")
    print(f"Decoded encodeQ: {tok.decode(q_ids)}")
