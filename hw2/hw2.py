import sys
import math
def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    list_char = {chr(key): 0 for key in range(ord('A'), ord('Z')+1)}
    with open (filename,encoding='utf-8') as f:
        for line in f: 
            characters_list = list(line)
            for char in characters_list:
                upperChar = char.upper()
                if upperChar >= "A" and upperChar <= "Z":
                    list_char[upperChar] +=1
    return list_char



# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
if __name__ == "__main__":
    fileName = "./letter.txt"
    print("Q1")
    char_list = shred(fileName)
    for char in char_list: 
        print(char + " " + str(char_list[char]))   

    print("Q2")
    es = {}
    with open("e.txt", 'r') as f:
        for line in f: 
            eng, prob = line.strip().split(" ")
            es[eng] = float(prob)
    number1 = char_list["A"] * math.log(es["A"])
    print(round(number1, 4))
    ss = {}
    with open("s.txt", 'r') as f:
        for line in f: 
            spa, prob = line.strip().split(" ")
            ss[spa] = float(prob)
    number2 = char_list["A"] * math.log(ss["A"])
    print(round(number2, 4))
    
    print("Q3")
    (e,s)=get_parameter_vectors()
    sum_eng = 0
    for i in range(26):
        sum_eng += char_list[chr(i+ord("A"))] * math.log(e[i])
    f_eng = math.log(0.6) + sum_eng
    format_f_eng = "{:.4f}".format(f_eng)
    print(format_f_eng)

    sum_spa = 0
    for i in range(26):
        sum_spa += char_list[chr(i+ord("A"))] * math.log(s[i])
    f_spa = math.log(0.4) + sum_spa
    format_f_spa = "{:.4f}".format(f_spa)
    print(format_f_spa)        
    
    print("Q4")
    conditional_pro = 0.0
    format_con = "{:.4f}".format(conditional_pro)
    if f_spa - f_eng >= 100:
        print(format_con)
    if f_spa - f_eng <= -100:
        print(float(format_con) + 1.0)
    else:
        conditional_pro = 1.0 / (math.exp(f_spa - f_eng))
        format_con2 = "{:.4f}".format(conditional_pro)
        print(format_con2)
