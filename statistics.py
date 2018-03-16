from collections import defaultdict
import re
import bz2, json

f = [w for w in
     open('korpus_cleaned.txt',encoding='utf-8').read().split()]
# f = open('korpus_cleaned.txt', encoding='utf-8')
voc = defaultdict(int)
capitals = re.compile('^[\u0410-\u042F\u04BA\u04E8\u0494\u04AE\u04A2\u0401]+$')  # capital symbols of yakutian char

for sentence in f:
    for word in sentence.strip().split():
        voc[word] += 1
print(voc)
words = list(voc.keys())
for word in words:
    if (re.match(capitals, word) and len(word) < 5) or len(word) > 26:
        del voc[word]
    else:
        word_l = word.lower()
        if word != word_l:
            voc[word_l] += voc[word]
            del voc[word]

words = list(voc.keys())
len(voc), len(words)
#print(len(voc), len(words))

# f = bz2.open('voc___.json.bz2', 'wb')

f = bz2.BZ2File('voc___.json.bz2', 'wb')
f.write(json.dumps(voc, ensure_ascii=False).encode(encoding='utf-8'))
f.close()
# download voc of counts
jsontext = bz2.BZ2File('voc.json.bz2', 'r').read().decode(encoding='utf-8')
voc = json.loads(jsontext)
del jsontext

words = list(voc.keys())
#print(len(words))#386233
#print (words[:10]) #[u'\u044d\u0442\u0438\u0438\u0442\u0438\u043d\u044d\u044d\u0495\u044d\u0440', u'\u043f\u043b\u0440\u043e\u0449\u0430\u0434\u043a\u0430', u'\u0431\u0430\u0442\u044b\u0439\u0430\u043d\u04
average_word_len = sum([len(w)*voc[w] for w in words])/sum([voc[w] for w in words])
word_count = sum([voc[k] for k in voc])
WORD_LEN_COEFF = 1
THRESHOLD_COEFF = 0.5
AFFIX_LEN = 1
f = open("korpus_cleaned.txt", encoding='utf-8')
sl = 0
nl = 0
for sentence in f:
    sl += len(sentence.strip().split())
    nl += 1
average_sent_len= sl/nl
print(average_sent_len)
from collections import defaultdict
prob = defaultdict(lambda: 0)
for word in voc:
    for char in word:
        prob[char]+=voc[word]
total = sum([n for n in prob.values()])#84263863
for k,v in prob.items():
    prob[k] = v/total
from matplotlib import pyplot as plt
import numpy as np
letters = list(prob.keys())
words=list(voc.keys())
len_word = sum([len(w)*voc[w] for w in words]) / word_count  #average word length
len_search = int(len_word * WORD_LEN_COEFF)
cond_prob = defaultdict(lambda: 0)
total = defaultdict(lambda: 0)

for word in voc:#для слова в словаре
    positions = range(-min(len_search, len(word) - 2), 0) # from -7 to 0
    for i in positions:
        cond_prob[(i, word[i])] += voc[word]
        total[i] += voc[word] # dictionary with prob of char words?
for posChar in cond_prob: #получаем из частот вероятности
    i = posChar[0]
    cond_prob[posChar] /= total[i]
thres_cond = defaultdict(lambda: 0.0)
maxlet = ['']*8
#для каждой позиции ищем букву с наибольшим значением условной вероятности,
#половина УВ данной буквы считается за порог медиального разбиения
for posChar in cond_prob:#цикл по позициям букв в условной вероятности
    i = posChar[0]
    if cond_prob[posChar] > thres_cond[i]:
        thres_cond[i] = cond_prob[posChar]
        maxlet[-i] = posChar[1]
for pos in thres_cond:
            thres_cond[pos] *= THRESHOLD_COEFF
thres_cond, maxlet
#порог медиального разбиения - половина условной вероятности , буквы с УВ не меньше порога - верхнее подмножеств
cond_prob_sup = {}
for posChar in cond_prob:
        i = posChar[0]
        if cond_prob[posChar] > thres_cond[i]:
            cond_prob_sup[posChar] = cond_prob[posChar]
#cond_prob_sup # верхнее подмножество
cf = {}
for posChar in cond_prob_sup:
    char = posChar[1]
    cf[posChar] = cond_prob_sup[posChar] / prob[char]
import operator
informants = {}
for pos in range(-len_search, 0):
    kmax = max({k for k in cf if k[0]==pos}, key=lambda k: cf[k])
    informants[pos] = (kmax[1], cf[kmax])
from collections import Counter
freq = [(word[-2], n) for word, n in voc.items() if len(word)>1 and word[-1]==informants[-1][0]]#частота буквы в -2 позиции перед буквой "н"
cond_prob_let = defaultdict(int) #словарь условных вероятностей символов
for let, n in freq:
    cond_prob_let[let] += n
cond_prob_let = sorted(cond_prob_let.items(), key=operator.itemgetter(1), reverse=True)#сортируем элементы словаря по вероятностям в убыв порядке

freq = [(word[-2], n) for word, n in voc.items() if len(word)>1 and word[-1]=='н']#частота буквы в -2 позиции перед буквой "н"
cond_prob_let = defaultdict(int) #словарь условных вероятностей символов
for let, n in freq:
    cond_prob_let[let] += n
cond_prob_let = sorted(cond_prob_let.items(), key=operator.itemgetter(1), reverse=True)#сортируем элементы словаря по вероятностям в убыв порядке
_end = '_end_'
root = {}
def make_trie(*words):
     for word in words:
         current_dict = root
         for letter in word:
             current_dict = current_dict.setdefault(letter, {})
         current_dict[_end] = _end
     return root
#for word in words[:10]:
#    print(make_trie(word))
w = words[:1000]
def build_trie(words):
    root = {'n':0} #кол-во букв встречающихся в
    for w in words: #для слова в списке
        word = w[::-1] # переворачиваем список, читаем слово с конца
        current_dict = root
        for letter in word: #для буквы в слове
            current_dict = current_dict.setdefault(letter, {'n': 0}) #получить значение из словаря по ключу.
            #Автоматически добавляет элемент словаря, если он отсутствует.
        current_dict['#'] = True
        current_dict = root
        for letter in word: #для буквы в словаре
            current_dict['n']+=voc[w]
            current_dict = current_dict.setdefault(letter, {'n': 0})
        current_dict['n']+=voc[w]
        current_dict['#'] = True
    return root
big_trie = build_trie(words)
#n_let = big_trie['н']#слова, заканчивающиеся на "н"
n_let = big_trie[informants[-1][0]]#слова, заканчивающиеся на "н"
L={letter: n_let[letter]['n'] for letter in n_let.keys() if letter!='n'}
from operator import itemgetter

sorted_x = sorted(L.items(), key=itemgetter(1))[::-1]

for i in range(len(sorted_x)):
        print(sorted_x[i][1]/sorted_x[i+1][1], end='\n')
        if (sorted_x[i][1]/sorted_x[i+1][1]>1.5):
            print (">1.5")
            aff=sorted_x[0][0]+informants[-1][0]
            print(aff)
            break
print(sorted_x[1][0])
