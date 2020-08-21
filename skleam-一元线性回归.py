from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv',delimiter=',')
x_data = data[:,0]# 表示取从第0行到最后1行，但是只需要第0列
y_data = data[:,1]# 表示取从第0行到最后1行，但是只需要第1列
plt.scatter(x_data,y_data)
plt.show()
print(x_data.shape)

x_data = data[:,0,np.newaxis]
y_data = data[:,1,np.newaxis]
# 创建并拟合模型
model = LinearRegression()# 实现线性回归的类
model.fit(x_data,y_data)

# 画图
plt.plot(x_data,y_data,'b.')
plt.plot(x_data,model.predict(x_data),'r')
plt.show()