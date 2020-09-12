# 使用SimpleTagBased算法对Delicious2K数据进行推荐
# 原始数据集：https://grouplens.org/datasets/hetrec-2011/
# 数据格式：userID     bookmarkID     tagID     timestamp
import random
import math
import operator
import pandas as pd


"""0.数据准备，变量定义"""
file_path = "./user_taggedbookmarks-timestamps.dat"

records = {}   # 从原始数据生成user->item->tag记录，保存了user对item的tag，即{userid: {item1:[tag1, tag2], ...}}

train_data = dict()   # 训练集
test_data = dict()   # 测试集

user_tags = dict()   # 用户user及其使用过tags和次数
tag_items = dict()   # 标签tag及其标记过item和次数
user_items = dict()   # 用户user及其标记过的item和次数
tag_users = dict()   # 标签tag及使用过它的user和次数
item_tags = dict()   # 商品item及标记过它的tag和次数
item_users = dict()   # 商品item及标记过它用户user和次数

"""1. 数据加载，生成records"""
def load_data():
    print("开始数据加载...")
    df = pd.read_csv(file_path, sep='\t')
    for i in range(len(df)):
        uid = df['userID'][i]
        iid = df['bookmarkID'][i]
        tag = df['tagID'][i]

        records.setdefault(uid,{})    # uid键不存在时，设置默认值{}
        records[uid].setdefault(iid,[])   # 嵌套字典，iid键不存在时，设置默认值[]
        records[uid][iid].append(tag)
        
    print("数据集大小为 %d." % (len(df)))
    print("设置tag的人数 %d." % (len(records)))
    print("数据加载完成！\n")
    
"""2. 数据处理，将数据集拆分为训练集和测试集"""
def train_test_split(ratio=0.2, seed=100):
    random.seed(seed)
    for u in records.keys():
        for i in records[u].keys():
            if random.random()<ratio:   # ratio比例设置为测试集
                test_data.setdefault(u,{})
                test_data[u].setdefault(i,[])
                for t in records[u][i]:
                    test_data[u][i].append(t)
            else:
                train_data.setdefault(u,{})
                train_data[u].setdefault(i,[])
                for t in records[u][i]:
                    train_data[u][i].append(t)        
    print("训练集样本数 %d, 测试集样本数 %d" % (len(train_data),len(test_data)))

"""3. 数据初始化，使用records生成user_tags, tag_items, user_items, tag_users, item_tags, item_users"""
# 设置矩阵 mat[index, item] = 1
def addValueToMat(mat, index, item, value=1):
    if index not in mat:
        mat.setdefault(index,{})
        mat[index].setdefault(item,value)
    else:
        if item not in mat[index]:
            mat[index][item] = value
        else:
            mat[index][item] += value

# 使用训练集，初始化user_tags, tag_items, user_items
def initStat():
    records=train_data
    for u,items in records.items():
        for i,tags in items.items():
            for tag in tags:
                addValueToMat(user_tags, u, tag, 1)
                addValueToMat(tag_items, tag, i, 1)
                addValueToMat(user_items, u, i, 1)
                addValueToMat(tag_users, tag, u, 1)
                addValueToMat(item_tags, i, tag, 1)
                addValueToMat(item_users, i, u, 1)
    print("user_tags, tag_items, user_items, tag_users, item_tags, item_users初始化完成.")
    print("user_tags大小 %d, tag_items大小 %d, user_items大小 %d" % (len(user_tags), len(tag_items), len(user_items)))
    print("tag_users大小 %d, item_tags大小 %d, item_users大小 %d" % (len(tag_users), len(item_tags), len(item_users)))

"""4. 生成推荐列表，Top-N"""
    # 对用户user推荐Top-N
def recommend(user, N, norm_type):
    recommend_items=dict()
    tagged_items = user_items[user]     
    for tag, wut in user_tags[user].items():
        for item, wti in tag_items[tag].items():
            if item in tagged_items:
                continue
            if norm_type == "SimpleTagBased":   # SimpleTagBased算法
                # 对于该user使用过的所有标签 [( user使用某一标签tag的次数 wut) * ( 所有用户使用该标签tag标记item的次数 wti )]
                norm = 1
            elif norm_type == "NormTagBased-1":   # NormTagBased-1算法：除以（这个tag被所用用户使用过的次数 * 这个user使用过tag的数量）
                norm = len(tag_users[tag].items()) * len(user_tags[user].items())
            elif norm_type == "NormTagBased-2":   # NormTagBased-2算法：除以（这个user使用过tag的数量 * 被这个tag标记过item的数量）
                norm = len(user_tags[user].items()) * len(tag_items[tag].items())
            elif norm_type == "TagBased-IDF":   #TagBased-IDF算法：
                norm = math.log(len(tag_users[tag].items())+1)
            else:
                print("norm_type参数错误！")
                break
                
            if item not in recommend_items:
                recommend_items[item] = wut * wti / norm
            else:
                recommend_items[item] = recommend_items[item] + wut * wti / norm
    return sorted(recommend_items.items(), key=operator.itemgetter(1), reverse=True)[0:N]

"""注：SimpleTagBased推荐分数计算过程"""
# user=8的推荐结果[(1416, 61), (1526, 50), (4535, 47), (4639, 46), (23964, 46)]
# 其中item = 1416的得分计算过程
#for key in user_tags[8]: 
#    wti = str(tag_items[key].get(1416,0))
#    if wti != '0':
#        wut = str(a[key]) 
#        print("str(key) + ":" + wut "+ "*" + wti)


"""5. 使用测试集，计算准确率和召回率"""
def precisionAndRecall(N, norm_type):
    hit = 0
    h_recall = 0
    h_precision = 0
    for user,items in test_data.items():
        if user not in train_data:
            continue
        # 获取Top-N推荐列表
        rank = recommend(user, N, norm_type)
        for item,rui in rank:
            if item in items:
                hit = hit + 1
        h_recall = h_recall + len(items)
        h_precision = h_precision + N
    #print('一共命中 %d 个, 一共推荐 %d 个, 用户设置tag总数 %d 个' %(hit, h_precision, h_recall))
    # 返回准确率 和 召回率
    return (hit/(h_precision*1.0)), (hit/(h_recall*1.0))

# 使用测试集，对推荐结果进行评估
def testRecommend(norm_type):
    print("推荐结果评估")
    print("%3s %10s %10s" % ('N',"精确率",'召回率'))
    for N in [5,10,20,40,60,80,100]:
        precision,recall = precisionAndRecall(N, norm_type)
        print("%3d %10.3f%% %10.3f%%" % (N, precision * 100, recall * 100))



"""--------------分割线--------------------"""
load_data()
train_test_split(ratio = 0.2)
initStat()
testRecommend(norm_type = "TagBased-IDF")   # norm_type{"SimpleTagBased", "NormTagBased-1", "NormTagBased-2", "TagBased-IDF"}
