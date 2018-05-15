import numpy as np
from tool import input_fn
from data import generate_digit_data
import random
import math
#a,b = input_fn(1, 1000, 'linear', 'uniform', False)
keys_list = generate_digit_data('load', 'uniform')[1][:50]
#print(keys_list)
#print(np.shape(keys_list))
#print(a)
#print(b)
def generater_ip_list(listLen):
    numList = []
    for i in range(listLen):
        num = random.randint(0,255)
        numList.append(num)
    ips = []

    for i in range(listLen):
        ip = "192.168.1.%d"%(numList[i])
        ips.append(ip)
    return ips
def save_hashList(hashList):
    f = open("data/hashList.txt",'w')
    num = 0
    for i in hashList:
        line = "%d,%s,%s\n"%(num,i[num][0],i[num][1])
        num += 1
        f.write(line)
    return
def save_ipList():
    return

def loadHashList():
    f = open("data/hashList.txt","r")
    hashList = []
    while(1):
        line  = f.readline()
        if(not line):
            break
        #print(line[:-1])
        a = line.split(",")
        location = int(a[0])
        key = a[1]
        ip = a[2]
        hashList.append([location,key,ip[:-1]])
    return hashList
"""
def load_ip():
    f = open("data/ips.txt")
    ips = []
    while(1):
        line = f.readline()
        if not line:
            break
        ips.append(line[:-1])
    return ips
"""
def dataHash(keys_list,hashList):
    for i in keys_list:
        location = hashFunc(i,100)
        #print("i = %f,location = %d"%(i,location))
        while(1):
            list = hashList[location][location]
            if(location>99):
                location = 0
            elif list[0]=="null":
                list[0] = i
                hashList[location][location] = list
                break
            else:
                location+=1
                if location>99:
                    break
                list = hashList[location][location]
    return hashList

def hashFunc(key,hashListLen):
    val = key * key * 26431
    val = val % hashListLen
    val = int(val)
    #print(val)
    return val

def hostHash(ips,hashList):
    for i in ips:
        LastipNum = i.split('.')[-1]
        LastipNum = int(LastipNum)
        location = hashFunc(LastipNum,100)

        #print ("ip = %s,location = %d"%(i,location))
        list = hashList[location][location]
        while(1):
            if(location>99):
               # print("location>99")
                location = 0
            elif list[1]=="null":
                list[1] = i
                hashList[location][location] = list
              #  print("%s isNull %s in list"%(i,list))
                break
            else:
             #   print("location +=1 %d"%(location))
                location+=1
                if location>99:
                    break
                list = hashList[location][location]

    return hashList
def spliHasgList(hashList):
    ipList = []
    dataList = []
    for i in hashList:
        if i[1] != "null":
            dataList.append([i[0],i[1]])
        if i[2] != "null":
            ipList.append([i[0],i[1]])
    return dataList,ipList

def allocateData(hashList):
    ipLoad = []
    tmpData = []
    for i in hashList:

        if (i[2]!="null"):
            ipLoad.append([i[2],tmpData])
            tmpData = []
        elif(i[1]!="null"):
            tmpData.append(i[1])
    return ipLoad
def flatIpLoad(ipload):
    for i in ipload:
        tmpData = []
        for j in i[1]:
            for t in j:
                tmpData.append([t])
        i[1] = tmpData
        tmpData = []

    return ipload


def allocateData2(hashList):
    ipLoad = []
    tmpData = []
    for i in hashList:
        if (i[2]!="null"):
            ipLoad.append([i[2],tmpData])
            #print(i[2])
            tmpData = []
        elif(np.shape(i[1])[0]!=0):
            #print(i[1])
            tmpData.append(i[1])
    return ipLoad

def static_ip_load(ipLoadList):
    for i in ipLoadList:
        #print(i[0])
        #a = np.reshape(i[1],[1])
        #print(np.shape(a))
        #print(i[1])
        line = "%s,%d"%(i[0],np.shape(i[1])[0])
        print(line)
    return



def init_hashList(listLen):
    hashList = []
    for i in range(listLen):
        dict = {i:['null','null']}
        hashList.append(dict)
    return hashList
#hashFunc(119,100)
#hashList = init_hashList(100)
##print(hashList)
##print(hashList[0])
#ips = load_ip()
#ips = ips[:10]
#hashList = hostHash(ips,hashList)
#hashList = dataHash(keys_list=keys_list,hashList=hashList)
#save_hashList(hashList)
#print(hashList)
#hashList =   loadHashList()
#print(hashList)
#hashList = np.array(hashList)
#print(hashList)
#ipload = allocateData(hashList)
#static_ip_load(ipload)
