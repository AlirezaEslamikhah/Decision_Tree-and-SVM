from numpy.core.fromnumeric import argmax
import pandas as pd
import numpy as np
import math
import copy
from matplotlib import pyplot 

class node:
    def __init__(self , attribute, table = None, Value = None, children = None, ig = None):
        self.attribute = attribute      # نام ویژگی
        self.table = None               # جدول داده ها
        self.Value = []                 # مقادیر ویژگی
        self.children = []              # فرزندان
        self.ig = None           # اطلاعات گرفته شده
    








def PluralityValue(examples , y):
    number = examples.shape[0]                      # تعداد مثال ها
    results = list(examples[y])                     # مقادیر متغیر هدف
    attr1 = results[0]                              # مقدار اول
    attr2 = results[1]                              # مقدار دوم
    count1 = 0                                      # تعداد مقدار اول
    count2 = 0                                      # تعداد مقدار دوم
    for i in range(number):                         # برای هر مثال 
        if results[i] == attr1:                     # اگر مقدار اول بود
            count1 += 1                             # تعداد اول را افزایش میدهیم
        elif results[i] == attr2:                   # اگر مقدار دوم بود
            attr2 = results[i]                      # مقدار دوم را تغییر میدهیم
            count2 += 1
    if count1 > count2: 
        return node(attr1)                          # اگر تعداد اول بیشتر بود یک گره با مقدار اول برمیگرداند
    else : 
        return node(attr2)                          # اگر تعداد دوم بیشتر بود یک گره با مقدار دوم برمیگرداند


def clasification(Examples , y):
    if Examples.empty:                              # اگر جدول خالی بود
        return                              
    length = Examples.shape[0]                              # تعداد مثال ها
    res = list(Examples[y])                                 # مقادیر متغیر هدف
    Value = res[0]                                          # مقدار اول
    for i in range(length):                                 # برای هر مثال
        if res[i] != Value:                                 # اگر مقدار متغیر هدف مثال متفاوت از مقدار اول بود
            return -1                                       # -1 برمیگرداند
    return res[0]                                           # مقدار اول را برمیگرداند

def Entropy(Example , attribute):
    c = Example[attribute]           # مقادیر ویژگی
    c , remainder = pd.factorize(c)  # تبدیل مقادیر به عدد
    Vk = np.bincount(c)              # تعداد مقادیر
    Entropy = 0                      # انتروپی
    pb = Vk / len(c)                 # احتمال هر مقدار
    for p in pb:                     # برای هر مقدار
        if p > 0 :
            Entropy = Entropy + p*np.log2(p)  # انتروپی را محاسبه میکنیم
    return -Entropy                           # انتروپی را برمیگردانیم

def ig(table , attribute , y):
    parent_entropy = Entropy(table , y)   #parent entropy 
    d = table[attribute].unique()         # مقادیر ویژگی
    Vk = list()                         # جدول های مقادیر ویژگی 
    for Value in d:                     # برای هر مقدار ویژگی
        Vk.append(table[table[attribute] == Value]) # جدول های مقادیر ویژگی را میسازیم
    Remainder = 0                       # مقدار باقی مانده
    for i in range(len(Vk)):            # برای هر جدول
        prob = Vk[i].shape[0]/table.shape[0]        # احتمال مقدار ویژگی
        Remainder = Remainder + prob*Entropy(Example=Vk[i] , attribute=y)   # مقدار باقی مانده را محاسبه میکنیم
    return parent_entropy - Remainder                                       # اینفورمیشن گین را برمیگردانیم

def Importance(X , table , y):
    igv = list()                                        # اینفورمیشن گین ها
    for attribute in X:                                 # برای هر ویژگی
        igv.append(ig(table , attribute , y))           # اینفورمیشن گین ها را محاسبه میکنیم
    IA = argmax(igv)                                    # ایندکس اینفورمیشن گین بیشترین
    infogainValue = max(igv)                            # اینفورمیشن گین بیشترین
    return X[IA] , infogainValue


def PrintTree(root , y):
    q = []  
    q1 = []
    q.append(root); 
    nodeC = 0
    nodeR = 0
    print("The root is "+str(root.attribute)+" and the IG is "+str(root.ig)+" and the Entropy is "+str(Entropy(root.table , y))+" and the number of examples is "+str(root.table.shape[0])+"\n")
    while (len(q) != 0):
        p = q[0]
        q.pop(0)
        for i in range(len(p.children)):
            nodeC = nodeC+1
            # if str(p.children[i].attribute) !='Yes' and str(p.children[i].attribute) !='No':
            print("The child is "+str(p.children[i].attribute)+" and the IG is "+str(p.children[i].ig)+" and the Entropy is "+str(Entropy(p.children[i].table , y))+" and the number of examples is "+str(p.children[i].table.shape[0])+"\n")
            q.append(p.children[i])
            q1.append(nodeC)
        if q1 == []:
            return 
        nodeR = q1[0]
        q1.pop(0)
    return 

        
def DecisionTree(table , attribute , examples , y , Atvalues , AttributeAll):
    if table.empty:                                                     # جدول خالی است
        return PluralityValue(examples , y)                             # مقدار بیشترین را برمیگردانیم  
    elif attribute == []:                                               # ویژگی خالی است
        return PluralityValue(examples , y)                             # مقدار بیشترین را برمیگردانیم
    elif clasification(table , y) != -1:                                # جدول یک کلاس است
        return node(clasification(table , y))                           # یک نود با مقدار کلاس برمیگردانیم
    else:
        V = list()                                                            # لیستی برای مقادیر ویژگی ها
        X = attribute                                                   # ویژگی ها را کپی میکنیم
        best_attribute , infogainValue = Importance(table=table , X=X , y=y)    # بهترین ویژگی را میگیریم
        Tree = node(best_attribute)                                             # یک نود با بهترین ویژگی میسازیم
        Tree.ig = infogainValue                                                 # اینفورمیشن گین را میذاریم
        Tree.table = table                                                      # جدول را میذاریم
        AttIndex = AttributeAll.index(best_attribute)                           # ایندکس بهترین ویژگی را میگیریم
        d = Atvalues[AttIndex]                                                  # مقادیر ویژگی را میگیریم
        for Value in d:                                                         # برای هر مقدار ویژگی
            Tree.Value.append(Value)                                            # مقدار را به لیست مقادیر اضافه میکنیم
            V.append(table[table[best_attribute] == Value])                   # جدول هایی که مقدار ویژگی برابر با مقدار ویژگی هست را به لیست اضافه میکنیم
        AT = copy.deepcopy(X)                                                   # کپی از ویژگی ها
        AT.remove(best_attribute)                                               # بهترین ویژگی را از ویژگی ها حذف میکنیم
        for Vk in V:
            Sub = DecisionTree(Vk , AT , table ,y , Atvalues , AttributeAll)    # با هر جدولی که مقدار ویژگی برابر با مقدار ویژگی هست رکرسیون میزنیم
            Sub.table = Vk                                                      # جدول را میذاریم
            Tree.children.append(Sub)                                           # نود را به لیست فرزندان اضافه میکنیم
        return Tree                                                             # نود را برمیگردانیم
    
    

if __name__ == '__main__':
    table = pd.read_csv('restaurant.csv')                   # خواندن از فایل csv
    X = list(table.columns)                                 # جدا کردن ستون ها
    y = X[-1]                                               # آخرین ستون را به عنوان خروجی می گیریم که در واقع هدف ماست
    X.remove(y)                                             # حذف آخرین ستون از ورودی ها چون هدف را به تنهایی داشتیم و دیگر نیازی به آن نیست
                                                            #در اینجا ما با پارامتری به نام ارزش در  گره ها سر و کار داریم
                                                            # این پارامتر به ما میگوید که در هر بار رسیدن به گره چه پارامتر هایی به سمت چپ و چه 
                                                            # پارامتر هایی به سمت راست باید برسند و اگر بیشتر از دو بچه داشتیم به کدام سمت هدایت شوند
    Atvalues = []
    for x in X:
        Atvalues.append(table[x].unique())                  # در اینجا ارزش هایی که متفاوت بود یا به اصطلاح تمامی مقادیر متفاوتی که هر ویژگی میتواند داشته باشد را 
                                                            # در یک لیست ذخیره میکنیم
    tree = DecisionTree(table, X,table,y,Atvalues,X)        # اینجا ما درخت تصمیم را با استفاده از تابع ما ایجاد میکنیم
    TreeResturant = PrintTree(tree , y=y)