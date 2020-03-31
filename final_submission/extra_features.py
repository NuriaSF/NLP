import numpy as np
import re
import scipy

def get_qlength(questions):
    qlen = []
    for quest in questions:
        clean_doc_pattern = re.compile( r"('\w)|([^a-zA-Z0-9.])") #Find words containing alphanumeric or points
        q = re.sub('\'s', '', quest) #Remove 's
        q = re.sub('\'t', ' not', q) #Change 't for not'
        q = re.sub('\'re', ' are', q) #Change 're for are'
        q = re.sub('[?%!@#$\'\""]', '', q)#Remove symbols
        q = re.sub('\.\s', ' ', q)#Remove points with a space afterwards
        clean_q = clean_doc_pattern.sub(" ", q)
        qlen.append(len(re.findall(r"(?u)\b[\w.,]+\b",q)))
        
    return np.array(qlen).reshape(-1,1)

def is_math(questions):
    math=[]
    for quest in questions:
        if '[math]' in quest:
            math.append(1)
        else:
            math.append(0)
    return np.array(math).reshape(-1,1)
    
def is_number(word):
    try :  
        w = float(word) 
        if(np.isnan(w)):
            return 0
        if(np.isinf(w)):
            return 0
        res = 1
    except : 
        res = 0
    return res    

def has_numbers(questions):
    num=np.zeros((len(questions)))
    which_num = np.zeros((len(questions)))
    i=0
    for quest in questions:
        for w in re.findall(r"(?u)\b[\w.,]+\b",quest):
            is_num = is_number(w)
            if is_num==1:
                num[i]=1
                which_num[i]=float(w)
                if(np.isnan(which_num[i])):
                    print(which_num[i])
                    print(float(w))
                break
        i+=1
    return num.reshape(-1,1), which_num.reshape(-1,1)


def is_different_number(which_num1, which_num2):
    dif = which_num1 - which_num2
    dif[dif>0]=1
    return np.array(dif).reshape(-1,1)

def q1_q2_intersect(row, q1, q2, q_dict):
    set1 = set(q_dict[q1[row]])
    set2 = set(q_dict[q2[row]])
    return(len(set1.intersection(set2))/len(set1.union(set2)))


def intersection(Xq1,Xq2):
    union = scipy.sum((Xq1!=0)+(Xq2!=0) ,axis=1)
    union[union==0]=1
    intersection = scipy.sum((Xq1!=0).multiply(Xq2!=0), axis=1)
    return intersection/union