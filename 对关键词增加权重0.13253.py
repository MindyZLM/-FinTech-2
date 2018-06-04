# coding=utf-8
import csv
import math
from string import punctuation
import re
import jieba

file_train_dir = "../train_data.csv"
file_test_dir = "../test_data_utf8.csv"

L2=['2784', '12057', '19620', '23768', '23814', '28091', '34581', '46930', '51579', '62560',
    '96556', '97000', '105587', '109969', '130887', '147430', '157489', '170901', '178244',
    '180411', '188481', '194912', '198100', '209500', '215415', '238209', '246035', '247997',
    '261419', '276555', '279555', '281603', '286775', '299355', '303297', '309796', '322697',
    '358093', '367410', '373991', '378273', '397450', '400484', '402272',
    '408519', '430604', '453004', '454748', '462743', '464543']
add_punc='，。、% 【】“”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=擅长于的&#@￥'
all_punc=punctuation+add_punc

jieba.add_word('地质勘探局')
jieba.add_word('地震勘探局')
jieba.add_word('地学研究中心')
jieba.add_word('美国MBA')
jieba.add_word('更多')
jieba.add_word('月报')
jieba.add_word('液化天然气')
jieba.add_word('持有量')
jieba.add_word('将会')
jieba.add_word('副总统')
jieba.add_word('雨宫正佳')
jieba.add_word('调至')
jieba.add_word('科夫尼')
jieba.add_word('并无')
jieba.add_word('地学研究中心')
jieba.add_word('科夫尼')
jieba.add_word('前总统')
jieba.add_word('经济再生大臣')
jieba.add_word('美元/桶')
jieba.add_word('朝鲜问题')
jieba.add_word('政治变化')
jieba.add_word('政策风险')
jieba.add_word('宽松政策')
jieba.add_word('GDPNowcast')
jieba.add_word('麦夸里岛')
jieba.add_word('就业市场')
jieba.add_word('副能源部长')

add_weight = ['地质勘探局','地震','贷款','液化天然气','原油','货币','欧洲央行','黄金','IMF','YTN','峰会',
              '朝鲜问题','核武器','陈茂波','政治变化','政策风险','物价','通缩','通胀','贸易战','美联储',
              '地学研究中心','无核化', '导弹','GDPNowcast','GDP','宽松政策','债券市场','原油','万桶','文在寅',
              '对话','美国财政部','拍卖','利率','EIA', '天然气','地缘','政治','投资','失业率','地学研究中心', 'GFZ',
              '脱欧','谈判','就业市场','地质勘探局','会晤','石油','革命','卫队','武装','弹道导弹', '通胀',
              '就业率','期货价格','访问','货币政策','PAJ','GDPNow','欧佩克', '加入','LME', '收涨', '伦铝和伦',
              '镍','袭击','停火', '冲突', '油价','两油','原油','WTI']


stop_words = ['年','月','日','对','就','将','和','是','至','.%','据','了']
# stop_words = ['年','月','对','就','将','和','是','至','.%','据','了']

def sentence_cut(x):#cut words and delete punctuation
    x=re.sub(r'[0-9]|d+.%','',x)#delet numbers and letters
    testline = jieba.cut(x,cut_all=False)
    testline=' '.join(testline)
    testline=testline.split(' ')
    te2=[]
    for i in testline:
        if i not in stop_words: #删去权重小的词
            te2.append(i)
            if i in all_punc:
                te2.remove(i)
    return te2


def cos_dist(a,b):
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1,b1 in zip(a,b):
        part_up += a1*b1  #计算分子
        # print(a1)
        # print(b1)
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq) #就算分母
    if part_down == 0.0:
        return 0
    else:
        return part_up/part_down

test_file = open(file_test_dir,encoding='utf-8')

test_reader = csv.reader(test_file)


for test_num, test_line in enumerate(test_reader):
    if test_num >= 1:
        a1 = test_line[1]
        a1 = sentence_cut(a1)  #test的字符串处理
        L = []
        d = {}
        # print(a1)

        train_file = open(file_train_dir, encoding='utf-8')
        train_reader = csv.reader(train_file)
        for train_num, train_line in enumerate(train_reader):
            if train_num >= 1:
                a2 = train_line[1]
                a2 = sentence_cut(a2)  # test的字符串处理
                # print(a2)

                b1 = {}   #存储test的分词结果
                c1 = []
                b2 = {}   #存储train的分词结果
                c2 = []
                # 数数,计算在a1出现字的次数，若a2中不存在则赋值0
                for i in a1:
                    if a1.count(i) > 0:
                        if i in add_weight:
                            b1[i] = a1.count(i) + 1
                        else:
                            b1[i] = a1.count(i)
                    if a2.count(i) > 0:
                        if i in add_weight:
                            b2[i] = a2.count(i) + 1
                        else:
                            b2[i] = a2.count(i)
                        # b2[i] = a1.count(i)
                    else:
                        b2[i] = 0
                # 计算在a2出现并且a1中不存在的字的次数，a1赋值0
                for i in a2:
                    if i not in a1:
                        if a2.count(i) > 0:
                            if i in add_weight: #判断是否在权重大的词中
                                b2[i] = a2.count(i) + 1
                            else:
                                b2[i] = a2.count(i)
                            # b2[i] = a2.count(i)
                            b1[i] = 0
                # print(b1)
                # print(b2)
                # break

                # 向量化,结果放入c1,c2中
                for i in b1.keys():
                    c1.append(b1[i])
                for i in b2.keys():
                    c2.append(b2[i])
                # print(c1)
                dist = cos_dist(c1, c2)
                # print(dist)
                L.append(dist)  # 往列表里面增加元素
                d[train_line[0]] = dist  # 往字典里面增加元素
        # 对列表、字典降序
        L = sorted(L, reverse=True)
        L = L[0:23:1]  # 提取最前面23个元素
        d = sorted(d.items(), key=lambda item: item[1], reverse=True)
        d = d[0:23:1]  # 提取最前面23个元素
        print(d)

        with open("test7.txt","a") as f:
            for k in range(21):
                if test_line[0]!=d[k][0]:
                    s = test_line[0] + '\t' + d[k][0] + '\n'
                    f.write(s)
                if d[k][0] in L2:
                    print(d[k][0])



