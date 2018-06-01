# -*- coding: utf-8 -*-

# ## Определения

from collections import defaultdict, Counter
from operator import itemgetter
import re,operator
import bz2, json
from collections import defaultdict
import math
WORD_LEN_COEFF = 1
THRESHOLD_COEFF = 0.5
DROP = 1.5
DROP_1 = 2
AFFIX_LEN = 1
THRESHOLD_OSTAT=0.5
N=5#шлейфовый порог

trie, voc, words, prob,word_count,average_word_len,bases, affixes, next_bases, false_affixes, removed_ost, specter\
    = None, None, None, None, None, None, None, None, None, None, None, None



def main():
    global trie, words, voc, prob, average_word_len
    voc = load_voc()
    words = sorted(list(voc.keys()))
    word_count = sum([voc[k] for k in voc])
    average_word_len = sum([len(w)*voc[w] for w in words]) / word_count
    len_search = int(average_word_len * WORD_LEN_COEFF) #это максимальная разрешенная длина аффикса
    print("{} словоформ, {} словоупотреблений, средняя длина слова {} ".format(len(words), word_count, average_word_len))

    # подсчет безусловных вероятностей букв
    # trie, prob = build_trie_and_prob(voc)
    prob = json.load(open("prob.json", encoding="utf-8"))
    strie = bz2.BZ2File('trie.json.bz2', 'r').read().decode(encoding='utf-8')
    trie = json.loads(strie)
    del strie
    print("Безусловные вероятности первых 10 букв:\n========================\n", 
        sorted([(letter,nv) for letter,nv in prob.items()], key=itemgetter(1), reverse=True)[:10])

    # подсчет условных вероятностей букв
    cond_prob = build_cond_prob(voc, prob, len_search)

    # информанты - это буквы с макс значением КФ в каждой позиции
    informants = find_informants(prob, cond_prob, len_search)
    print("ИНФОРМАНТЫ:\n===================")
    print(informants)

    # отправной аффикс начинаем строить с информанта имеющего max КФ
    affix=informants[0]
    affix=extend_right(*affix)
    affix=extend_left(affix, trie, len_search)
    print("ОТПРАВНОЙ АФФИКС:\n===================")    
    print(affix)


def load_voc():
    # Загрузить словарь количеств из файла.
    # Словарь содержит частоты слов в виде {слово: число вхождений в корпус, ... },
    # например {"көппөҕү" : 4, "хазар" : 3, ...}
    #
    # корпус в формате txt  занимет 174 МБ, словарь частот в json 10,4МБ,
    # после сжатия в формат .bz2 1,7 МБ
    svoc = bz2.BZ2File('voc.json.bz2', 'r').read().decode(encoding='utf-8')
    voc = json.loads(svoc)
    del svoc
    return voc


def build_trie_and_prob(voc):
    #подсчитываем частоты букв и строим дерево оконочаний
    prob = defaultdict(lambda: 0)
    trie = {'n':0}
    for w,n in voc.items(): #для каждого слова в списке
        word = w[::-1]  # переворачиваем слово, читаем слово с конца
        current_dict = trie
        trie['n'] += n
        for letter in word:  # для буквы в слове
            prob[letter]+=n
            current_dict = current_dict.setdefault(letter, {'n': 0}) #получить значение из словаря по ключу.
                                                                     #Автоматически добавляет элемент словаря, если он отсутствует.
            current_dict['n']+=n
        current_dict['#'] = n

    total = sum([n for n in prob.values()])#84263863
    for k,v in prob.items():
        prob[k] = v/total

    return trie, prob


def build_cond_prob(voc, prob, len_search):
    letters = list(prob.keys())

    cond_prob = defaultdict(lambda: 0) #словарь для условных вероятностей
    total = defaultdict(lambda: 0)

    for word,n in voc.items():#для слова в словаре
        positions = range(-min(len_search, len(word) - 2), 0) # from -7 to 0
        for i in positions:
            cond_prob[(i, word[i])] += n
            total[i] += n # dictionary with prob of char words?

    for posChar in cond_prob: #получаем из частот вероятности
        i = posChar[0]
        cond_prob[posChar] /= total[i]

    return cond_prob


def find_informants(prob, cond_prob, len_search):
    max_cond = defaultdict(lambda: 0.0)
    maxlet = ['']*8
    #для каждой позиции ищем букву с наибольшим значением условной вероятности,
    for posChar in cond_prob:#цикл по позициям букв в условной вероятности
        aff_len = posChar[0]
        if cond_prob[posChar] > max_cond[aff_len]:
            max_cond[aff_len] = cond_prob[posChar]
            maxlet[-aff_len] = posChar[1]

    print("Наиболее частые буквы по позициям:\n============================\n", maxlet[-1:0:-1],"\n")

    print("Максимальные вероятности по позициям:\n============================\n", max_cond,"\n")
    #порог медиального разбиения - половина условной вероятности , буквы с УВ не меньше порога - верхнее подмножеств
    cond_prob_sup = {}
    for posChar in cond_prob:
            i = posChar[0]
            if cond_prob[posChar] > THRESHOLD_COEFF * max_cond[i]:
                cond_prob_sup[posChar] = cond_prob[posChar]

    # КФ = условная вер по данной позиции / безусл вероятность
    cf = {}
    for posChar in cond_prob_sup:
        char = posChar[1]
        cf[posChar] = cond_prob_sup[posChar] / prob[char]

    print("КФ для верхних подмножества:\n====================\n");
    for aff_len in set(map(itemgetter(0), cf.keys())):
        print(aff_len, "**")
        for k,v in cf.items():
            if k[0] == aff_len:
                print(k[1], "{:.4f}".format(v), end="  ")
        print("")

    # информанты - это буквы с макс значением КФ в каждой позиции
    informants = []
    for aff_len in range(-len_search, 0):
        kmax = max({k for k in cf if k[0] == aff_len}, key=lambda k: cf[k])
        informants.append((kmax[1], aff_len, cf[kmax]))

    informants.sort(key = itemgetter(2), reverse=True)
    return informants


def extend_right(char, pos, cf):
    if pos == -1:#если информант в последней позиции, то расширять некуда
        return char #возвращаем информант как аффикс
    d = defaultdict(int)
    for w,n in voc.items():#для буквы и частоты в словаре
        if w[pos:pos+1]==char: #если буква в позиции равна нашей, то посчитаем это окончание
            d[w[pos+1:]]+=n
    return char+max(d.keys(), key=lambda end: d[end]) #прибавляем к информанту самое частое окончание


def extend_left(affix, trie, len_search):
    #расширяем аффикс влево используя trie

    current_dict = trie
    for ch in affix[::-1]:
        current_dict = current_dict[ch]

    aff_len = len(affix)

    """ 
    Для поиска буквы слева:
        идем по дереву trie
        по две самые частотные буквы делим друг на друга, при мере перепада большей 1.5 прибавляем к информанту более частую из них.
       Иначе начинаем рассматривать по две самые частотные буквы/на следующие две, 
    если мера перепада в одной из них больше двух, то из данной пары берем более частотную и прибавляем ее к аффиксу. 
    """
    #пока позиция символа в слове больше разрешенной длины аффикса
    while aff_len < len_search:
        #составляем список всех букв предшествующих аффиксу с количествами
        L = [(l, current_dict[l]["n"]) for l in current_dict.keys() if l not in '#n']
        #сортируем по количествам
        L.sort(key = itemgetter(1), reverse = True)
        #if affix=='нан':
            #import pdb
            #pdb.set_trace()
        ch = L[0][0]
        if L[0][1] > DROP*L[1][1]:
            affix = ch + affix
            current_dict = current_dict[ch]
        else:
            if (L[0][1]+L[1][1]) / (L[2][1]+L[3][1]) > 2:
                affix = ch + affix
                current_dict = current_dict[ch]
            else:
                break
        aff_len+=1

    return affix


main()

affix=[extend_left(extend_right('л', -5, 1.94903926924671), trie, 7)]
print(affix[0])

# узел trie, соответствующийокончанию aff
def affix_node(aff):
    global trie
    current_node = trie
    for char in aff[::-1]:
        current_node = current_node[char]
    return current_node

# рекурсивно возвращает все основы, растущие из данного узла
def word_dfs(node, ending=''):
    result = [ending] if '#' in node else []
    for ch in node:
        if ch in ['#', 'n']: continue
        result += word_dfs(node[ch], ch+ending)
    return result
  
# все основы, растущие из данного узла 
def bases_with_affix(aff):
    return sorted([b for b in word_dfs(affix_node(aff)) if len(b)>1])

from bisect import bisect_left
# суммарная встречаемость основы b с любыми остатками
def build_freq_bases(b):
    freq = 0
    for w in words[bisect_left(words, b):]:
                if not w.startswith(b) : break   
                freq += voc[w]
    return freq   


def build_ost(bases):
    global words, voc
    ostat = defaultdict(int)
    for b in bases:
        affix_pos = len(b)
        for w in words[bisect_left(words, b):]:
            if not w.startswith(b): break 
            if not w[affix_pos:] in affix:
                ostat[w[affix_pos:]] += voc[w]  # вариант с подсчетом словоупотреблений
                # ostat[w[affix_pos:]] += 1  # вариант с подсчетом словоформ    
    return ostat

#affixes - нормальные


def fast_alg(bases, specter, freq_bases):
    global removed_ost
    max_ost_val=max(specter.values())
    #те пары к в у которых к больше макс
    inf_zveno={ost: v for ost,v in specter.items() if v > max_ost_val / 2}
    print("Zveno: ", inf_zveno)
    #дольше нужна сочетаемость с некоторой группой контрольных основ
    #верхнее подмножество баз очередного вида    
    next_base_freq={}
    max_nb_freq=0
    
    freq_cur_bases = {b: sum([voc.get(b+ost,0) for ost in specter]) for b in bases}
    max_freq_cur = max(freq_cur_bases.values())
    print("Max freq: ",max_freq_cur)
    #верхнее подмножество баз очередного вида
    control_bases=[b for b,freq in freq_cur_bases.items() if freq >= max_freq_cur/2 ]
    if len(control_bases)==1:
        lower=[(b, freq)for b,freq in freq_cur_bases.items()if freq<max_freq_cur]
        control_bases.append(max(lower, key=itemgetter(1))[0])
    print("Control bases", control_bases)
    
    #Первый критерий принадлежности к парадигме - сочетаемость остатков в звене с основами control_bases
    keep_ost = [ost for ost in inf_zveno if all([b+ost in voc for b in control_bases])]
    removed_ost = [ost for ost in inf_zveno if ost not in keep_ost]  
    
    print("Keep:", keep_ost)
    #destiny_of_affix(false_affixes,next_bases,voc)
    print("!!!!!!!!!!!Removed odt:", removed_ost)
    return keep_ost

def destiny_of_affix(removed_ost, specter, next_bases, voc):
    #проверка на меру децентрации
    #если >=1/2 синтагматической вероятности падает на парадигматически малую(0,1) часть баз - то аффикс искл из парадигмы до конца рассм
    #иначе - аффикс выводится из звена, но сохраняется в спектре остатков
    aff_sochet=defaultdict(int),
    sintagm_removed_aff=defaultdict(int)
    for aff in removed_ost:
        for base in next_bases:
            if aff+base in voc.items():
                aff_sochet[aff]+=1
        #freq = 0
    for aff in removed_ost:
        for w in words[bisect_left(words),aff:]:
            if not w.endswith(aff): break
            sintagm_removed_aff[aff] += voc[w]
    len_next_bases= len(next_bases)
    zv = [aff for aff,freq in sintagm_removed_aff.items() if 1/2*freq>=0.1*len(next_bases)]
    #if
    #проверка сочетаемости синтагматической аероятности аффикса с количеством оставшихся после групповой редукции баз, принимающих данный аффикс

    specter.append(affix)
    return sintagm_removed_aff


def direct_alg(bases, specter, false_affixes):
    global prob, voc
    # верхнее подмножество остатков текущего вида
    m = max(specter.values()) * THRESHOLD_OSTAT
    upper_set = {ost: val for ost,val in specter.items() if val > m }
    if not [ost for ost in upper_set if ost not in false_affixes]:
        sp_list = sorted([(ost,val) for ost,val in specter.items()], key=itemgetter(1), reverse=True)
        for ost,val in sp_list:
            if ost not in false_affixes:
                break
        upper_set[ost] = val
    print("Верхнее подмножество остатков текущего вида,", len(upper_set), "шт.") 
    # ВЫЧИСЛИТЬ незав  для остатков из upper_set
    nv = {}
    summ_kol = 0

    for ost, kol in specter.items(): #ostat - defdict
        summ_kol += kol
        nezav_ver = 1
        for ch in ost:
            nezav_ver *= prob[ch]
        nv[ost]=nezav_ver

    # усл вероятности
    uv = {}
    for ost, kol in specter.items():
        uv[ost] = kol / summ_kol
    # КФ - отношение условной вероятности к безусловной
    corr_func={}
    for ost in upper_set:
        corr_func[ost]= uv[ost]/nv[ost]
    corr_func = [(ost, cf) for ost,cf in corr_func.items() if ost not in false_affixes]
    corr_func = sorted(corr_func, key = itemgetter(1), reverse= True)
    print("Коррелятивная функция: ", corr_func)
    if not corr_func:  # суперпороговые редукции исчерпали спектр остатков
        print("Остались только ложные остатки ")
        return []
    # найти след информант
    informant = corr_func[0][0]
    print("Информант: ", informant)
    return [informant]

def check_agglut_part():
    return 0


def build_class(bootstrap_affix):
    global words, average_word_len,THRESHOLD_OSTAT, thres_reduction, false_affixes, specter

    k = 10**(math.log(average_word_len, 10) / (1 + 0.02*math.log( len(voc), 10)) )
        # коэффициент редукции
    print("Поправочный коэффициент:", k)
    thres_reduction = 1 / average_word_len  #порог редукции
    print("Порог редукции:", thres_reduction)

    affixes = [bootstrap_affix]  # найденные аффиксы парадигмы
    false_affixes=[]  # список отвергнутых аффиксов, давших  ложный шаг

    bases = [bases_with_affix(bootstrap_affix)]
    specter = [build_ost(bases[0])]
    freq_bases = {b: build_freq_bases(b) for b in bases[0]}
    step = 1
    fast = False

    while True:
        print("\n***", step)
        print("Основы {}-го вида: {} шт.".format(step, len(bases[-1])))
        print("Аффиксы ",affixes)
        print("Спектр остатков {}-го вида: {} шт.".format(step, len(specter[-1])))

        if not specter[-1]:  # исчерпаны все остатки в спектре
            print("Исчерпаны все остатки в спектре")
            break         
        if not fast:
            print("Прямой ход: ")
            next_affixes = direct_alg(bases[-1], specter[-1], false_affixes)
            if not next_affixes:
                break
        else:
            print("Ускоренный ХОД!!!!!!!!")
            next_affixes = fast_alg(bases[-1], specter[-1], freq_bases)


        #основы следующего вида
        next_bases = [b for b in bases[-1] if all([b+aff in voc for aff in next_affixes]) and
                                             freq_bases[b]>step]
          
        # Поправочный коэффициент
        # увеличивается во столько раз сколько аффиксов было сохранено в звене
        K = k*len(next_affixes)

        # Мера редукции
        # доля основ текущего вида, не принимающих остатки следующего вида
        N = len(bases[-1])
        print("K: ", K)
        reduction = (N - len(next_bases)) / (K * N)
        print("Мера редукции: ", reduction)
        
        if reduction > thres_reduction: #суперпороговая редукция
            false_affixes+=next_affixes
            print("Суперпороговая редукция, ложные остатки", false_affixes)
            if len(false_affixes)>average_word_len:
                print("Cуперпороговая редукция повторяется большее число раз, чем средняя длина словоформы")
                break
        else:
            step += 1
            false_affixes = []
            affixes += next_affixes
            #спектр остатков следующего вида
            next_specter = {ost: sum([voc.get(b + ost, 0 ) for b in next_bases]) for ost in specter[-1]}
            next_specter = {ost:v for ost, v in next_specter.items() if v>0 and ost not in next_affixes}

            bases.append(next_bases)
            specter.append(next_specter)

            if (len(next_bases)<=2):
                print("Остались две базы очередного вида") 
                break
            if reduction < thres_reduction/2:#если редукция < порога редукции/2(порог устойчивости)
                fast=True
            else:                        
                fast=False

    return bases[-1], affixes
#find_informants()
#destiny_of_affix(removed_ost,specter, next_bases,voc)
def rostpriny(removed_ost):
    print ("Proverka na removed",removed_ost)

bases, affixes = build_class(affix[0])
rostpriny(removed_ost)
print("bases and affixes: ",bases, affixes)
THRESHOLD_OSTAT=0.5


