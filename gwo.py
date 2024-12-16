import numpy as np
import random
import copy
from sklearn import metrics
import warnings
import pandas as pd

''' 种群初始化函数 '''


def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = int(random.random() * (ub[j] - lb[j]) + lb[j])

    return X, lb, ub


'''边界检查函数'''


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] < lb[j]:
                X[i, j] = lb[j]
            elif X[i, j] > ub[j]:
                X[i, j] = ub[j]
    return X


'''计算适应度函数'''


def CaculateFitness(X, fun, train_X, vali_X, train_Y, vali_Y):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :], train_X, vali_X, train_Y, vali_Y)
    return fitness


'''适应度排序'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''根据适应度对位置进行排序'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


'''灰狼算法'''


def GWO(pop, dim, lb, ub, MaxIter, fun, df1,df2,model):
    Alpha_pos = np.zeros([1, dim])
    Alpha_score = float("-inf")
    Beta_pos = np.ones([1, dim])
    Beta_score = float("-inf")
    Delta_pos = np.ones([1, dim])
    Delta_score = float("-inf")


    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun, df1,df2,model)  # 计算适应度值
    indexBest = np.argmax(fitness)
    GbestScore = copy.copy(fitness[indexBest])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[indexBest, :])
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):

        for i in range(pop):
            fitValue = fun(X[i, :], df1,df2,model)
            if fitValue > Alpha_score:
                Alpha_score = copy.copy(fitValue)
                Alpha_pos[0, :] = copy.copy(X[i, :])

            if fitValue < Alpha_score and fitValue > Beta_score:
                Beta_score = copy.copy(fitValue)
                Beta_pos[0, :] = copy.copy(X[i, :])

            if fitValue < Alpha_score and fitValue < Beta_score and fitValue > Delta_score:
                Delta_score = copy.copy(fitValue)
                Delta_pos[0, :] = copy.copy(X[i, :])

        a = 2 - t * (2 / MaxIter)
        for i in range(pop):
            for j in range(dim):
                r1 = random.random()
                r2 = random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = np.abs(C1 * Alpha_pos[0, j] - X[i, j])
                X1 = Alpha_pos[0, j] - A1 * D_alpha

                r1 = random.random()
                r2 = random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = np.abs(C2 * Beta_pos[0, j] - X[i, j])
                X2 = Beta_pos[0, j] - A2 * D_beta

                r1 = random.random()
                r2 = random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_beta = np.abs(C3 * Delta_pos[0, j] - X[i, j])
                X3 = Delta_pos[0, j] - A3 * D_beta

                X[i, j] = (X1 + X2 + X3) / 3

        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun, df1,df2,model)  # 计算适应度值
        indexBest = np.argmax(fitness)
        if fitness[indexBest] >= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitness[indexBest])
            GbestPositon[0, :] = copy.copy(X[indexBest, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve


def fun1(X, train_X, vali_X, train_Y, vali_Y):
    model = RandomForestRegressor(n_estimators=int(X[0]),max_depth=int(X[1]),max_features=int(X[2]),
                                  min_samples_split=int(X[3]), min_samples_leaf=int(X[4])
                                  )

    model.fit(train_X, train_Y)
    Y_pred = model.predict(vali_X)
    # 通过F1值作为评价指标
    r2_score_res = r2_score(vali_Y, Y_pred)

    return r2_score_res


'''主函数 '''

# 选择数据集
data_file = 'D:/船舶油耗预测/处理后结果.csv'

# 读取数据
data = pd.read_csv(data_file)
data = data.dropna(axis=0)
# 划分数据集

train_X,test_X,train_Y,test_Y = train_test_split(data.iloc[:,:-1],data.iloc[:,-1],train_size=0.7,random_state=1)

# 设置参数
pop = 10  # 种群数量
MaxIter = 5  # 最大迭代次数
dim = 5  # 维度
lb = np.array([100, 1, 1,2,1])  # 下边界
ub = np.array([300, 20, 8,20,20])  # 上边界
# 选择适应度函数
fobj = fun1
# 灰狼算法
GbestScore, GbestPositon, Curve = GWO(pop, dim, lb, ub, MaxIter, fobj, train_X, test_X, train_Y, test_Y)
print('最好的参数为：{}'.format(GbestPositon))
best_model = RandomForestRegressor(n_estimators=int(GbestPositon[0][0]),max_depth=int(GbestPositon[0][1]),max_features=int(GbestPositon[0][2]),
                                   min_samples_split=int(GbestPositon[0][3]),min_samples_leaf=int(GbestPositon[0][4]))
best_model.fit(train_X, train_Y)
Y_pred = best_model.predict(test_X)

r2_sco = r2_score(test_Y, Y_pred)
mae = mean_absolute_error(test_Y, Y_pred)
mse = mean_squared_error(test_Y, Y_pred)

print('测试集R2为：{}'.format(r2_sco))
print('测试集MSE为：{}'.format(mse))
print('测试集MAE为：{}'.format(mae))



Curve = Curve[:, 0].tolist()

x = [i for i in range(len(Curve))]
# 绘制图像
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(x, Curve, label='r2', alpha=0.5, color='red', linewidth=2, marker='+',
        linestyle='--')

ax.set_xlabel('index')
ax.set_ylabel('r2')

plt.savefig('./迭代图.png', dpi=600)
plt.show()




