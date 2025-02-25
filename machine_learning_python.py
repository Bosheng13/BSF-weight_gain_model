#!/usr/bin/env python
# coding: utf-8


####raw data analysis
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
warnings.filterwarnings("ignore")


###load weight gain data######

data = pd.read_excel('d:/scaled-weight.xlsx')
data
data.info()



data.describe()

########visualize by violin plot#######
import seaborn as sns
column = list(data.columns)[:-1]
fig, ax =plt.subplots(2,3,constrained_layout=True, figsize=(8,6))
colors = ['tan','darkorange','cyan','#7FFFD4','m','#FFE4C4','#FFEBCD','#0000FF','#8A2BE2','#A52A2A','#DEB887','#5F9EA0',
          '#7FFF00','#F08080','#FFE4C4','r']
a = 1
b = 0
for i in range(len(column)):
    if b!=0 and b%3 == 0:
        b = 0
        a += 1
    pic = sns.violinplot(data[column[i]].tolist(),ax=ax[a-1,b],color=colors[i],width=0.6)
    pic.set_title(column[i])
    b+=1
fig.savefig('d:/violin-plot.png', dpi=600)


#####check the distribution of loaded data#######
fig, ax =plt.subplots(1,1,constrained_layout=True, figsize=(6,4))
sns.kdeplot(data['Weight gain'], color='b', shade=True)
fig.savefig('d:/distribution.png', dpi=600)


########correlation analysis#######
corr = data.corr()

##############visualization of correlation matrix###########
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Pearson Correlation Matrix')
plt.savefig('d:/correlation.png', dpi=600)
plt.show()


########Spliting data,70% for train data set; 30% for test data set############

from sklearn.model_selection import train_test_split
train_X,test_X,train_Y,test_Y = train_test_split(data.iloc[:,:-1],data.iloc[:,-1],train_size=0.8,random_state=14)
columns = train_X.columns.to_list()

##########Check the data distribution in the train and test data set##############
fig,ax = plt.subplots(2,3,figsize=(10, 8))
for i in range(2):
    for j in range(3):
        pic = sns.kdeplot(train_X[columns[i*3+j]], shade=True, color="r",ax=ax[i,j])
        pic = sns.kdeplot(test_X[columns[i*3+j]], shade=True, color="g",ax=ax[i,j])
        pic.set_xlabel(columns[i*3+j],fontsize=10)

fig.savefig('d:/Trainvstest-distribution.png', dpi=600)


##########################################
###########Decision tree model############
##########################################
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
######choose the best parameter###
param_grid = { 'max_depth':np.arange(1,10,1),
              'max_features':np.arange(1,10,1),
             'min_samples_split':np.arange(1,10,1),
              'min_samples_leaf':np.arange(1,10,1),
             }

rfr = DecisionTreeRegressor(random_state=1)
reg = GridSearchCV(rfr,
                   param_grid,
                   scoring='r2',
                   cv=3,verbose = 1).fit(train_X, train_Y)

best_alpha = reg.best_params_
best_score = reg.best_score_

best_alpha

best_score

##building Decision tree model##########
dt_model = DecisionTreeRegressor(random_state=1,max_depth=9,max_features=1,min_samples_leaf=1,min_samples_split=2)
dt_model.fit(train_X, train_Y)
y_pred_dt = dt_model.predict(test_X)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print('决策树模型——R2：{}'.format(r2_score(test_Y,y_pred_dt)))
print('决策树模型——RMSE：{}'.format(np.sqrt(mean_squared_error(test_Y,y_pred_dt))))
print('决策树模型——MAE：{}'.format(mean_absolute_error(test_Y,y_pred_dt)))

x = [i for i in range(len(y_pred_dt))]
#####visualiation####
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(x, y_pred_dt, label='预测值', alpha=0.5, color='blue', linewidth=1, marker='o',
        linestyle='--')
ax[0].plot(x, test_Y.tolist(), label='真实值', alpha=0.5, color='red', linewidth=2, marker='+',
        linestyle='--')
ax[0].legend()
ax[0].set_xlabel('测试集实例')
ax[0].set_ylabel('Weight gain')

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(np.array(test_Y.tolist()).reshape(-1,1),np.array(y_pred_dt.tolist()).reshape(-1,1))
ax[1].scatter(test_Y.tolist(),y_pred_dt.tolist(), color = 'red')
ax[1].plot(np.array(test_Y.tolist()), lr.predict(np.array(test_Y.tolist()).reshape(-1,1)),color ='blue')

ax[1].text(min(test_Y.tolist()), max(y_pred_dt.tolist()), f'R2 = {r2_score(test_Y, y_pred_dt):.3f}')
ax[1].text(min(test_Y.tolist()), max(y_pred_dt.tolist())-0.003, f'RMSE = {np.sqrt(mean_squared_error(test_Y,y_pred_dt)):.3f}', size=10,)
ax[1].text(min(test_Y.tolist()), max(y_pred_dt.tolist())-0.006, f'MAE = {mean_absolute_error(test_Y,y_pred_dt):.3f}', size=10,)
ax[1].text(min(test_Y.tolist()), max(y_pred_dt.tolist())-0.03, 'Y = {}*X+{}'.format(round(lr.coef_[0][0],3),round(lr.intercept_[0],3) , size=10,))
ax[1].set_xlabel('真实值')
ax[1].set_ylabel('预测值')
plt.savefig('d:/增量预测/预测结果图_决策树.png')
plt.show()




##########################################
###########Random Forest##################
##########################################

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
###choose the best parameter###
param_grid = { 'max_features':np.arange(1,7,1),
             }
rfr = RandomForestRegressor(random_state=1,n_estimators=110,max_depth=13,max_features=6)
reg = GridSearchCV(rfr,
                   param_grid,
                   scoring='r2',
                   cv=3,verbose = 1).fit(train_X, train_Y)

best_alpha = reg.best_params_
best_score = reg.best_score_

best_alpha

best_score


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=1,n_estimators=110,max_depth=13,max_features=6)
model.fit(train_X, train_Y)
rf_pre = model.predict(test_X)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print('随机森林模型——R2：{}'.format(r2_score(test_Y,rf_pre)))
print('随机森林模型——RMSE：{}'.format(np.sqrt(mean_squared_error(test_Y,rf_pre))))
print('随机森林模型——MAE：{}'.format(mean_absolute_error(test_Y,rf_pre)))


########visualization########
x = [i for i in range(len(rf_pre))]
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(x, rf_pre, label='预测值', alpha=0.5, color='blue', linewidth=1, marker='o',
        linestyle='--')
ax[0].plot(x, test_Y.tolist(), label='真实值', alpha=0.5, color='red', linewidth=2, marker='+',
        linestyle='--')
ax[0].legend()
ax[0].set_xlabel('测试集实例')
ax[0].set_ylabel('Weight gain')

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(np.array(test_Y.tolist()).reshape(-1,1),np.array(rf_pre.tolist()).reshape(-1,1))
ax[1].scatter(test_Y.tolist(),rf_pre.tolist(), color = 'red')
ax[1].plot(np.array(test_Y.tolist()), lr.predict(np.array(test_Y.tolist()).reshape(-1,1)),color ='blue')

ax[1].text(min(test_Y.tolist()), max(rf_pre.tolist()), f'R2 = {r2_score(test_Y, rf_pre):.3f}')
ax[1].text(min(test_Y.tolist()), max(rf_pre.tolist())-0.003, f'RMSE = {np.sqrt(mean_squared_error(test_Y,rf_pre)):.3f}', size=10,)
ax[1].text(min(test_Y.tolist()), max(rf_pre.tolist())-0.006, f'MAE = {mean_absolute_error(test_Y,rf_pre):.3f}', size=10,)
ax[1].text(min(test_Y.tolist()), max(rf_pre.tolist())-0.03, 'Y = {}*X+{}'.format(round(lr.coef_[0][0],3),round(lr.intercept_[0],3) , size=10,))
ax[1].set_xlabel('真实值')
ax[1].set_ylabel('预测值')
plt.savefig('d:/RF.png')
plt.show()


##########################################
####################SVR###################
##########################################
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

######choose the best parameter###
param_grid = {'C':np.arange(1,200,10),
              'epsilon':np.arange(0,1,0.1),
             }

rfr = SVR(random_state=1)
reg = GridSearchCV(rfr,
                   param_grid,
                   scoring='r2',
                   cv=3,verbose = 1).fit(train_X, train_Y)

best_alpha = reg.best_params_
best_score = reg.best_score_

best_alpha

best_score

svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_model.fit(train_X, train_Y)
y_pred = svr_model.predict(test_X)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print('SVR模型——R2：{}'.format(r2_score(test_Y,y_pred)))
print('SVR模型——RMSE：{}'.format(np.sqrt(mean_squared_error(test_Y,y_pred))))
print('SVR模型——MAE：{}'.format(mean_absolute_error(test_Y,y_pred)))

x = [i for i in range(len(y_pred))]
# 绘制图像
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(x, y_pred, label='Predicted', alpha=0.5, color='blue', linewidth=1, marker='o',
        linestyle='--')
ax[0].plot(x, test_Y.tolist(), label='Truth', alpha=0.5, color='red', linewidth=2, marker='+',
        linestyle='--')
ax[0].legend()
ax[0].set_xlabel('Training size')
ax[0].set_ylabel('Weight gain')

####visualization#######
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(np.array(test_Y.tolist()).reshape(-1,1),np.array(y_pred.tolist()).reshape(-1,1))
ax[1].scatter(test_Y.tolist(),y_pred.tolist(), color = 'red')
ax[1].plot(np.array(test_Y.tolist()), lr.predict(np.array(test_Y.tolist()).reshape(-1,1)),color ='blue')

ax[1].text(min(test_Y.tolist()), max(y_pred.tolist()), f'R2 = {r2_score(test_Y, y_pred):.3f}')
ax[1].text(min(test_Y.tolist()), max(y_pred.tolist())-0.003, f'RMSE = {np.sqrt(mean_squared_error(test_Y,y_pred)):.3f}', size=10,)
ax[1].text(min(test_Y.tolist()), max(y_pred.tolist())-0.006, f'MAE = {mean_absolute_error(test_Y,y_pred):.3f}', size=10,)
ax[1].text(min(test_Y.tolist()), max(y_pred.tolist())-0.009, 'Y = {}*X+{}'.format(round(lr.coef_[0][0],3),round(lr.intercept_[0],3) , size=10,))
ax[1].set_xlabel('Ground truth (Weight gain, gram)')
ax[1].set_ylabel('Predicted (Weight gain, gram)')
plt.savefig('d:/SVR.svg',format='svg')
plt.show()

##########################################
#############Linearregression#############
##########################################

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


model = LinearRegression()
model.fit(train_X, train_Y)
lr_pred = model.predict(test_X)

print('Linear regression model——R2：{}'.format(r2_score(test_Y,lr_pred)))
print('Linear regression model——RMSE：{}'.format(np.sqrt(mean_squared_error(test_Y,lr_pred))))
print('Linear regression model——MAE：{}'.format(mean_absolute_error(test_Y,lr_pred)))

#####Visualization#########
x = [i for i in range(len(lr_pred))]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(x, lr_pred, label='Predicted', alpha=0.5, color='blue', linewidth=1, marker='o',
        linestyle='--')
ax[0].plot(x, test_Y.tolist(), label='True', alpha=0.5, color='red', linewidth=2, marker='+',
        linestyle='--')
ax[0].legend()
ax[0].set_xlabel('Training size')
ax[0].set_ylabel('Weight gain (gram)')

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(np.array(test_Y.tolist()).reshape(-1,1),np.array(lr_pred.tolist()).reshape(-1,1))
ax[1].scatter(test_Y.tolist(),lr_pred.tolist(), color = 'red')
ax[1].plot(np.array(test_Y.tolist()), lr.predict(np.array(test_Y.tolist()).reshape(-1,1)),color ='blue')

ax[1].text(min(test_Y.tolist()), max(lr_pred.tolist()), f'R2 = {r2_score(test_Y, lr_pred):.3f}')
ax[1].text(min(test_Y.tolist()), max(lr_pred.tolist())-0.007, f'RMSE = {np.sqrt(mean_squared_error(test_Y,lr_pred)):.3f}', size=10,)
ax[1].text(min(test_Y.tolist()), max(lr_pred.tolist())-0.014, f'MAE = {mean_absolute_error(test_Y,lr_pred):.3f}', size=10,)
ax[1].text(min(test_Y.tolist()), max(lr_pred.tolist())-0.021, 'Y = {}*X+{}'.format(round(lr.coef_[0][0],3),round(lr.intercept_[0],3) , size=10,))
ax[1].set_xlabel('Ground truth (Weight gain, gram)')
ax[1].set_ylabel('Predicted (Weight gain, gram)')
plt.savefig('d:/RF.svg',format='svg')
plt.show()



from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV




##########################################
###################XGBoost################
##########################################
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

param_grid = {'max_depth': np.arange(2,20,1),
       'learning_rate':[ 0.1,0.3,0.5,],
              'n_estimators':np.arange(10,200,10)
        }


xgb = XGBRegressor(nthread=10,random_state=1)
reg = GridSearchCV(xgb,
                   param_grid,
                   scoring='r2',
                   cv=3,verbose = 1).fit(train_X,train_Y)

best_alpha = reg.best_params_
best_score = reg.best_score_

best_alpha

best_score


xgb = XGBRegressor(nthread=10,random_state=1,learning_rate= 0.5, max_depth= 3, n_estimators=140)
xgb.fit(train_X, train_Y)
predict1 = xgb.predict(test_X)
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
print('XGBOOST模型——R2：{}'.format(r2_score(test_Y,predict1)))
print('XGBOOST模型——RMSE：{}'.format(np.sqrt(mean_squared_error(test_Y,predict1))))
print('XGBOOST模型——MAE：{}'.format(mean_absolute_error(test_Y,predict1)))

####Visualization#########
x = [i for i in range(len(rf_pre))]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(x, predict1, label='预测值', alpha=0.5, color='blue', linewidth=1, marker='o',
        linestyle='--')
ax[0].plot(x, test_Y.tolist(), label='真实值', alpha=0.5, color='red', linewidth=2, marker='+',
        linestyle='--')
ax[0].legend()
ax[0].set_xlabel('测试集实例')
ax[0].set_ylabel('Weight gain')

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(np.array(test_Y.tolist()).reshape(-1,1),np.array(predict1.tolist()).reshape(-1,1))
ax[1].scatter(test_Y.tolist(),predict1.tolist(), color = 'red')
ax[1].plot(np.array(test_Y.tolist()), lr.predict(np.array(test_Y.tolist()).reshape(-1,1)),color ='blue')

ax[1].text(min(test_Y.tolist()), max(predict1.tolist()), 'R2 = 0.808',)
ax[1].text(min(test_Y.tolist()), max(predict1.tolist())-0.01, 'RMSE = 0.022', size=10,)
ax[1].text(min(test_Y.tolist()), max(predict1.tolist())-0.02, 'MAE = 0.018', size=10,)
ax[1].text(min(test_Y.tolist()), max(predict1.tolist())-0.03, 'Y = {}*X+{}'.format(round(lr.coef_[0][0],3),round(lr.intercept_[0],3) , size=10,))
ax[1].set_xlabel('真实值')
ax[1].set_ylabel('预测值')
plt.savefig('d:/增量预测/预测结果图_xgboost.png')
plt.show()

###save model#####
import joblib
joblib.dump(xgb, 'd:/增量预测/model.pkl')


train_X.columns


#Abstract and generation of plots###these part was also analyzed in R programiming 
import shap
background_adult = shap.maskers.Independent(train_X, max_samples=100)
explainer = shap.Explainer(xgb.predict, background_adult)
shap_values = explainer(train_X)

shap.summary_plot(shap_values, train_X,show=False)
plt.savefig('d:/Absrtact.png')


shap.summary_plot(shap_values, train_X, plot_type="bar",show=False)
plt.savefig('d:/importance.png')


shap.force_plot(shap_values[0,:], train_X.iloc[0,:], matplotlib=True,show=False)
plt.savefig('d:/增量预测/single learn.png')




