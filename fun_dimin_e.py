import numpy as np
import time
import random
from tqdm import tqdm as tqdm_progress
from data_preprocess import parse_mnist

def init(idx, Ci, dir):
    data_images = parse_mnist(minst_file_addr="./train-images-idx3-ubyte.gz", flatten=True)
    Phi = np.load(f'./{dir}/Phi.npy')
    s = np.load(f'./{dir}/s.npy')

    u = s[idx].reshape(-1, 1)
    # 生成观测矩阵A
    seed = 1
    np.random.seed(seed)
    A = np.random.random((100, 784))
    A_ = A @ Phi

    # 设置参数
    n = 10  # batch size = 10
    p, q = A_.shape
    x0 = np.zeros((100, 1))    

    # 生成噪声
    seed =1
    np.random.seed(seed)
    r = np.random.random((p,1))
    r_i = np.random.random((int(p/n),1))

    # 生成样本数据
    Y = (A @ data_images[idx] / 255).reshape(-1, 1)
    a = np.zeros((n, int(p/n), q))
    y = np.zeros((n, int(p/n), 1))
    for k in range(n):
        a[k] = A_[int(k*p/n) : int((k+1)*p/n), :]
        y[k] = Y.reshape(-1, 1)[int(k*p/n) : int((k+1)*p/n), :]

    return u, x0, n, A_, Y, a, y, Ci, r, r_i



# 定义软阈值算子
def prox(x, lambd):
    for i in range(len(x)):
        if np.abs(x[i]) > lambd:
            x[i] = np.sign(x[i]) * (np.abs(x[i]) - lambd)
        else:
            x[i] = 0
    return x

# 定义 PG 算法：将A_整体作为输入，对堆叠数据A_和Y，执行一个梯度下降步长和一个近端算子
def PG(u, x0, n, A_, Y, Ci, r, lambd=0.00005, step=0.000001, T=10000):
    xt = x0
    
    s_diff = np.zeros(T)  # 存储差值的数组
    time_array = np.zeros(T)
    start_time = time.time()  
    
    for t in tqdm_progress(range(T), desc='Running PG', unit='iteration'):
        norm_squared = np.linalg.norm(xt - u, ord=2)**2
        len_u = len(u)
        s_diff[t] = (1/len_u) * norm_squared
        
        Y_t = Y + Ci * r ** t

        g0 = np.dot(xt.T, np.dot(A_.T, A_)) - np.dot(Y_t.T, A_)  # 光滑项梯度
        xt1 = xt - step * g0.T
        xt = prox(xt1, lambd)  # 近端算子

        end_time = time.time() 
        total_time = end_time - start_time
        time_array[t] = total_time
    return xt, s_diff, time_array

# 定义 B_PG 算法：固定（a，y），依次执行一个梯度下降和一个近端算子
def B_PG(u, x0, n, a, y, Ci, r_i, lambd=0.00001, step=0.000001, T=10000):
    xt = x0
    s_diff = np.zeros(T)
    time_array = np.zeros(T)
    start_time = time.time()
    for t in tqdm_progress(range(T), desc='Running B_PG', unit='iteration'):
        xti = xt
        norm_squared = np.linalg.norm(xt - u, ord=2)**2
        len_u = len(u)
        s_diff[t] = (1/len_u) * norm_squared
        
        for i in range(n):  # n个(y,A)
            y_t = y[i] + Ci * r_i ** t
            g0 = np.dot(xti.T, np.dot(a[i].T, a[i])) - np.dot(y_t.T, a[i])  # 光滑项梯度
            xti = xti - step * g0.T
            xti = prox(xti, lambd)  # 近端算子
        xt = xti
        
        end_time = time.time()
        total_time = end_time - start_time
        time_array[t] = total_time
    return xt, s_diff, time_array

# 定义 SPG 算法：随机选取一个（a，y）执行一个梯度下降和一个近端算子
def SPG(u, x0, n, a, y, Ci, r_i, lambd=0.00001, step=0.00001, T=10000):
    xt = x0
    
    s_diff = np.zeros(T)
    time_array = np.zeros(T)
    start_time = time.time()
    for t in tqdm_progress(range(T), desc='Running SPG', unit='iteration'):
        xti = xt
        norm_squared = np.linalg.norm(xt - u, ord=2)**2
        len_u = len(u)
        s_diff[t] = (1/len_u) * norm_squared
        for i in range(n):  # n个(y,A)
            r = random.randint(0, n-1)
            y_t = y[r] + Ci * r_i ** t
            g0 = np.dot(xti.T, np.dot(a[r].T, a[r])) - np.dot(y_t.T, a[r])  # 光滑项梯度
            xti = xti - step * g0.T
            xti = prox(xti, lambd)  # 近端算子
        xt = xti
        
        end_time = time.time() 
        total_time = end_time - start_time
        time_array[t] = total_time
    return xt, s_diff, time_array

# 定义 ADMM 算法：将A_整体作为输入，对堆叠数据A_和Y，传统ADMM进行一次原始和对偶更新
def ADMM(u, x0, n, A_, Y, Ci, r, lambd=0.1, rho=0.5, T=100):
    q = len(x0)
    x = x0
    z = np.zeros_like(x0)
    w = np.random.rand(q, 1)
    s_diff = np.zeros(T)
    time_array = np.zeros(T)
    start_time = time.time()
    for t in tqdm_progress(range(T), desc='Running ADMM', unit='iteration'):
        norm_squared = np.linalg.norm(x - u, ord=2)**2
        len_u = len(u)
        s_diff[t] = (1/len_u) * norm_squared
        
        Y_t = Y + Ci * r ** t
        
        x = np.dot(np.linalg.inv(np.dot(A_.T, A_) + rho * np.identity(q)), (np.dot(A_.T, Y_t).reshape(-1, 1) + rho * (z - w)))
        z = prox(x + w, lambd / rho)
        w = w + x - z
        
        end_time = time.time()
        total_time = end_time - start_time
        time_array[t] = total_time
    return x, s_diff, time_array

# 定义 B_ADMM 算法：固定（a，y），每对依次对传统ADMM进行一次原始和对偶更新
def B_ADMM(u, x0, n, a, y, Ci, r_i, lambd=0.00001, rho=0.5, T=10000):
    x = x0
    z = x
    q = len(x)
    w = np.random.rand(q, 1)  # 初始化w
    E = np.identity(q)  # 初始化E
    s_diff = np.zeros(T)
    time_array = np.zeros(T)
    start_time = time.time()
    for t in tqdm_progress(range(T), desc='Running B_ADMM', unit='iteration'):
        norm_squared = np.linalg.norm(x - u, ord=2)**2
        len_u = len(u)
        s_diff[t] = (1/len_u) * norm_squared
        for i in range(n):
            y_t = y[i] + Ci * r_i ** t
            x = np.dot(np.linalg.inv(np.dot(a[i].T, a[i]) + rho * E), (np.dot(a[i].T, y_t) + rho * (z - w)))
            z = prox(x + w, lambd / rho)
            w = w + x - z
        
        end_time = time.time()
        total_time = end_time - start_time
        time_array[t] = total_time
    return x, s_diff, time_array

# 定义PGRR算法: 随机排列（a，y），依次执行梯度下降，每次迭代只执行一个近端算子
def PGRR(u, x0, n, a, y, Ci, r_i, lambd=0.0001, step=0.00001, T=10000):
    xt = x0
    s_diff = np.zeros(T)
    time_array = np.zeros(T)
    start_time = time.time()
    for t in tqdm_progress(range(T), desc='Running PGRR', unit='iteration'):          # 迭代T次
        xti = xt
        norm_squared = np.linalg.norm(xt - u, ord=2)**2
        len_u = len(u)
        s_diff[t] = (1/len_u) * norm_squared
        # (A,y)随机排列
        x = np.arange(0,n)
        random.shuffle(x)
        a_ = np.zeros_like(a)
        y_ = np.zeros_like(y)
        for m in range(len(a)):
            a_[m] = a[x[m]]
            y_[m] = y[x[m]]      
        for i in range(n):      # n个(y,A)
            y_t = y_[i] + Ci * r_i ** t
            g0 = np.dot(xti.T, np.dot(a_[i].T,a_[i])) - np.dot(y_t.T,a_[i])   # 光滑项梯度
            xti = xti - step * g0.T
        xt = prox(xti, lambd)
        
        end_time = time.time()
        total_time = end_time - start_time
        time_array[t] = total_time
    return xt, s_diff, time_array
