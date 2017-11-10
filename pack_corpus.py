from collections import defaultdict, Counter
f = open('korpus_cleaned.txt', encoding='utf-8')
i = 0
c = Counter
voc = defaultdict(int)
for sentence in f:
    words = sentence.strip().split()
    for w in words:
        voc[w] += 1
