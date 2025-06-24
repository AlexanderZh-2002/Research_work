import pandas as pd
import numpy as np
import random
import scipy
import matplotlib.pyplot as plt


def function_creator(type_, args):
    if type_ == "normal":
        def func(x):
            res = 2
            while res < 0 or res > 1:
                res = random.normalvariate(mu = args[0], sigma = args[1])
            return res
    if type_ == "weibull":
        def func(x):
            res = 2
            while res < 0 or res > 1:
                res = random.weibullvariate(args[0], args[1])
            return res
    if type_ == "chi-square":
        def func(x):
            res = 2
            while res < 0 or res > 1:
                res = 0
                for j in range(args[0]):
                    res += random.normalvariate(0,1)**2
                res*= args[1]
            return res
    if type_ == "lognormal":
        def func(x):
            res = 2
            while res < 0 or res > 1:
                res = random.lognormvariate(mu = args[0], sigma = args[1])
            return res
    return func

def create_noize(df, noize_percentage):
    num = len(df)
    default_num = len(df[df["default"] == 1])
    for i in range(0, default_num):
        if random.randint(1,100) < noize_percentage:
            i2 = random.randint(default_num, num-1)
            while df["default"][i2] != 0:
                i2 = random.randint(default_num, num-1)
            df["default"][i2] = 1
            df["default"][i] = 0
    return df


def draw_hist(df, dict_):
    default_df = df[df["default"] == 1]
    n = 1
    for key in dict_.keys():
        if key != "default":
            fig = plt.figure(n, figsize=(5, 5))
            s1 = fig.add_subplot(2,1,1)
            df[key].plot(kind = 'hist')
            plt.xlabel(key)
            plt.ylabel("all")
            #s2 = fig.add_subplot(2,1,2)
            s2 = fig.add_subplot(2,1,2, sharex = s1, sharey = s1)
            default_df[key].plot(kind = 'hist')
            plt.xlabel(key)
            plt.ylabel("default")
            n+=1
            plt.show()
        



def create_df(dict_, num, default_percentage):
    frames = []
    def_num = int(num/100*default_percentage)
    res = pd.DataFrame(index = range(num), columns = ["default"])
    res["default"][0:def_num] = 1
    res["default"][def_num:] = 0
    for key, item in dict_.items():
        df = pd.DataFrame(index = range(num), columns = [key]+["default"])
        func = function_creator(item[0], item[1])
        #print(func)
        df[key] = df.apply(func, axis = 1)
        #print(df)
        if item[2][0] == "treshold_good":
            df.sort_values(by = [key], ascending =True, inplace = True, ignore_index = True)
            df["default"][0:def_num] = 1
            df["default"][def_num:] = 0
            #print("yes\n",df)
        if item[2][0] == "treshold_bad":
            df.sort_values(by = [key], ascending =False, inplace = True, ignore_index = True)
            df["default"][0:def_num] = 1
            df["default"][def_num:] = 0
        if item[2][0] == "uniform":
            df["default"][0:def_num] = 1
            df["default"][def_num:] = 0
        if item[2][0] == "concentrated":
            df.sort_values(by = [key], ascending = True, inplace = True, ignore_index = True)
            idx1 = list(df.index[df[key] >= item[2][1] - 0.1])
            idx2 = list(df.index[df[key] <= item[2][1] + 0.1])
            idx3 = list(set(idx1)&set(idx2))
            #print(idx3)
            idx = idx3[len(idx3)//2]
            #print(idx)
            if idx < def_num//2:
                df["default"][0:def_num] = 1
                df["default"][def_num:] = 0
            elif(idx > num - def_num//2):
                df["default"][num - def_num:] = 1
                df["default"][def_num:] = 0
            else:
                df.sort_values(by = [key], ascending = True, inplace = True, ignore_index = True)
                df["default"][0:idx - def_num//2] = 0
                df["default"][idx - def_num//2: idx + def_num//2] = 1
                df["default"][idx + def_num//2:] = 0
            df.sort_values(by = ["default"], ascending = False, inplace = True, ignore_index = True)
            #print("concentrated", len(df[df["default"]==1]))
        frames.append(df)
        #print("df\n", df)
        res = res.join(df[key])
    #print("\nresult\n", res)
    return res

dict1 = {
    "norm":["normal",[0.5,0.25],("treshold_good",)],
    "weib  k = 1.5":["weibull",[0.5,1.5],("treshold_good",)],
    "weib  k = 1":["weibull",[0.25,1],("treshold_good",)],
    "log":["lognormal",[0.5,0.25],("treshold_good",)],
    "chi":["chi-square",[3, 0.1],("treshold_good",)]
    }
dict2 = {
    "norm":["normal",[0.5,0.25],("treshold_good",)],
    "weib  k = 1.5":["weibull",[0.5,1.5],("treshold_bad",)],
    "weib  k = 1":["weibull",[0.25,1],("treshold_good",)],
    "log":["lognormal",[0.5,0.25],("treshold_bad",)],
    "chi":["chi-square",[3, 0.1],("treshold_bad",)]
    }
dict3 = {
    "norm":["normal",[0.5,0.25],("treshold_good",)],
    "weib  k = 1.5":["weibull",[0.5,1.5],("concentrated", 0.3)],
    "weib  k = 1":["weibull",[0.25,1],("treshold_good",)],
    "log":["lognormal",[0.5,0.25],("concentrated",0.6)],
    "chi":["chi-square",[3, 0.1],("concentrated", 0.5)]
    }
dict4 = {
    "norm":["normal",[0.5,0.25],("treshold_good",)],
    "weib  k = 1.5":["weibull",[0.5,1.5],("treshold_good",)],
    "weib  k = 1":["weibull",[0.25,1],("concentrated", 0.6)],
    "log":["lognormal",[0.5,0.25],("treshold_good",)],
    "chi":["chi-square",[3, 0.1],("treshold_bad",)]
    }

num = 10000
test_num = 2000
"""
default_percentage = 50
train_df = create_df(dict_, num, default_percentage)
test_df = create_df(dict_, test_num, default_percentage)

print("in result train:", len(train_df[train_df["default"]==1]), "test:", len(test_df[test_df["default"]==1]))
#draw_hist(df, dict_)
train_df = create_noize(train_df, 10)
draw_hist(train_df, dict_)
test_df = create_noize(test_df, 10)
#draw_hist(test_df, dict_)

train_df.to_csv("train_ds_april.csv", index = False)
test_df.to_csv("test_ds_april.csv", index = False)
"""

percentage = [25, 50, 75]
for i in range(3):
    print("PERCENTAGE", i)
    default_percentage = percentage[i]
    print("DF_1")
    train_df_1 = create_df(dict1, num, default_percentage)
    test_df_1 = create_df(dict1, test_num, default_percentage)
    print("DF_2")
    train_df_2 = create_df(dict2, num, default_percentage)
    test_df_2 = create_df(dict2, test_num, default_percentage)
    print("DF_3")
    train_df_3 = create_df(dict3, num, default_percentage)
    test_df_3 = create_df(dict3, test_num, default_percentage)
    #print("DF_4")
    #train_df_4 = create_df(dict4, num, default_percentage)
    #test_df_4 = create_df(dict4, test_num, default_percentage)
    
    train_df_1 = create_noize(train_df_1, 10)
    train_df_2 = create_noize(train_df_2, 10)
    train_df_3 = create_noize(train_df_3, 10)
    #train_df_4 = create_noize(train_df_4, 10)

    test_df_1 = create_noize(test_df_1, 10)
    test_df_2 = create_noize(test_df_2, 10)
    test_df_3 = create_noize(test_df_3, 10)
    #test_df_4 = create_noize(test_df_4, 10)
    
    train_df_1.to_csv("data/train1_"+str(default_percentage)+".csv", index = False)
    test_df_1.to_csv("data/test1_"+str(default_percentage)+".csv", index = False)
    train_df_2.to_csv("data/train2_"+str(default_percentage)+".csv", index = False)
    test_df_2.to_csv("data/test2_"+str(default_percentage)+".csv", index = False)
    train_df_3.to_csv("data/train3_"+str(default_percentage)+".csv", index = False)
    test_df_3.to_csv("data/test3_"+str(default_percentage)+".csv", index = False)
    #train_df_4.to_csv("data/train4_ds_april_"+str(default_percentage)+".csv", index = False)
    #test_df_4.to_csv("data/test4_ds_april_"+str(default_percentage)+".csv", index = False)

    
