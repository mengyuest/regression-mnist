import numpy as np
import numpy.matlib
from config import cf
import time
import matplotlib.pyplot as plt
from mnist import MNIST

class ExperimentResult:
    def __init__(self):
        self.train_loss=[]
        self.valid_loss=[]
        self.test_loss=[]
        self.train_err=[]
        self.valid_err=[]
        self.test_err=[]
        self.best_train_err = 0
        self.best_valid_err = 0
        self.best_test_err = 0
        self.weight_value = 0

def load_data():

    t1=time.time()

    if cf.load_from_raw:

        mndata = MNIST('data')
        images, labels = mndata.load_training()

        train_images=np.array(images[:cf.train_size])
        train_images=train_images/255.0
        train_images=np.concatenate(
            (np.ones((np.shape(train_images)[0], 1)), train_images), axis=1)#.astype(int)
        train_labels=np.array(labels[:cf.train_size])

        timages, tlabels = mndata.load_testing()

        if cf.first2k:
            test_images = np.array(timages[:cf.test_size])
        else:
            test_images = np.array(timages[cf.test_size*-1:])
        test_images = test_images/255.0
        test_images = np.concatenate(
            (np.ones((np.shape(test_images)[0], 1)), test_images), axis=1)#.astype(int)
        if cf.first2k:
            test_labels = np.array(tlabels[:cf.test_size])
        else:
            test_labels = np.array(tlabels[cf.test_size*-1:])

    else:
        train_images=np.loadtxt('data/train_images.txt')
        train_labels=np.loadtxt('data/train_labels.txt')
        test_images=np.loadtxt('data/test_images.txt')
        test_labels=np.loadtxt('data/test_labels.txt')

    t2=time.time()
    print("load data",'{0:.4f}'.format(t2-t1), 'seconds')
    return train_images,train_labels,test_images,test_labels


def shuffle(train_images,train_labels):
    np.random.seed(cf.seed)
    order_list = [x for x in range(np.shape(train_images)[0])]
    np.random.shuffle(order_list)
    train_images = train_images[order_list, :]
    train_labels = train_labels[order_list]
    return train_images,train_labels

def logistic_preprocessing(train_images, train_labels, test_images, test_labels):
    c1=cf.select_num1
    c2=cf.select_num2

    tr_img=train_images[np.where(np.logical_or(train_labels==c1,train_labels==c2))]
    tr_lbe=train_labels[np.where(np.logical_or(train_labels==c1,train_labels==c2))]
    tr_lbe=np.array(tr_lbe==c1).astype(int)

    te_img=test_images[np.where(np.logical_or(test_labels==c1,test_labels==c2))]
    te_lbe=test_labels[np.where(np.logical_or(test_labels==c1,test_labels==c2))]
    te_lbe=np.array(te_lbe==c1).astype(int)
    divide = (int)(np.shape(tr_img)[0] * cf.train_ratio)

    return tr_img[:divide,:],tr_lbe[:divide],\
           tr_img[divide:,:],tr_lbe[divide:],\
           te_img,te_lbe


#TODO Can we find better method to explore solution space for optimization?
def gradient_descent_method():
    a=1

"""
Logistic Regression
"""
def logistic_regression(total_train_images,total_train_labels,total_test_images,total_test_labels,order=0):

    train_images,train_labels,valid_images,valid_labels,test_images,test_labels = \
        logistic_preprocessing(total_train_images, total_train_labels,
                               total_test_images, total_test_labels)

    n = np.shape(train_images)[0]
    m = np.shape(train_images)[1]

    eta=cf.lg_eta
    error_cnt=0
    error_tol=cf.lg_error_tol

    prev_err=100
    t=0
    T=cf.lg_T
    w=np.zeros((1,m))[0]
    bestw=np.zeros((1,m))[0]
    best_tr_err=0
    best_va_err=0
    best_te_err=0

    res=ExperimentResult()

    train_y, _, _ = logistic_pred(w, train_images, train_labels)

    while error_cnt<error_tol:
        t1=time.time()
        eta=eta/(1+t/T)

        term=regularization(w,order)
        w-=eta*term

        delta=np.dot((train_labels-train_y),train_images)
        w+=eta*delta/n

        train_y, _, train_err = logistic_pred(w, train_images, train_labels)
        valid_y, _, valid_err=logistic_pred(w,valid_images,valid_labels)
        test_y, _, test_err=logistic_pred(w,test_images,test_labels)

        if valid_err < prev_err:
            error_cnt=0
            bestw=np.array(w)
            best_tr_err=train_err
            best_va_err=valid_err
            best_te_err=test_err
        else:
            error_cnt+=1
        prev_err=valid_err
        t+=1

        train_loss=logistic_loss(train_y,w,train_images,train_labels,order)
        valid_loss=logistic_loss(valid_y,w,valid_images,valid_labels,order)
        test_loss=logistic_loss(test_y,w,test_images,test_labels,order)

        t2 = time.time()

        res.train_loss.append(train_loss)
        res.valid_loss.append(valid_loss)
        res.test_loss.append(test_loss)
        res.train_err.append(train_err)
        res.valid_err.append(valid_err)
        res.test_err.append(test_err)

        if cf.debug:
            print("Epoch",t,"train_err","{0:.2%}".format(train_err),
                  "valid_err","{0:.2%}".format(valid_err),"test_err","{0:.2%}".format(test_err),
                  "cnt",error_cnt,"dt","{0:.4f}".format(t2-t1))
            print("eta", eta, "norm", np.linalg.norm(term), np.linalg.norm(eta * delta / n))
            print()

    print("Final   train_err", "{0:.2%}  ".format(best_tr_err),
          "valid_err","{0:.2%}  ".format(best_va_err),
          "test_err","{0:.2%}".format(best_te_err))
    print()

    weightPlot(bestw,cf.lg_header+"_"+str(cf.select_num1)+"_"+str(cf.select_num2))
    res.weight_value=bestw
    res.best_train_err=best_tr_err
    res.best_valid_err=best_va_err
    res.best_test_err=best_te_err
    return res


def logistic_loss(yy,w,images,labels,order):
    n=np.shape(images)[0]
    y=np.array(yy)
    omy = 1 - y
    y[y==0]=cf.very_small_number
    omy[omy==0]=cf.very_small_number
    E=np.sum(labels*np.log(y)+(1-labels)*np.log(omy))

    if order==1:
        return -E/n+cf.lg_lamda*np.sum(np.abs(w))
    elif order==2:
        return -E/n+cf.lg_lamda*(np.linalg.norm(w)**2)
    return -E/n

def regularization(w, order):
    n=np.shape(w)[0]
    term=np.zeros((1,n))[0]
    if order == 2:
        term=2*cf.lg_lamda*w
    elif order == 1:
        term=((w>=0)*2-1)*cf.lg_lamda
    return term

def logistic_pred(w,images,labels):
    n=np.shape(labels)[0]
    #TODO overflow problem
    mid=-np.dot(images,w)
    mid[mid>100]=100
    result=1/(1+np.exp(mid))
    preds_bin=np.array(result>=0.5).astype(int)
    error_rate = np.sum(preds_bin!=labels)/n
    return result, preds_bin, error_rate

def softmax_preprocessing(total_train_images, total_train_labels,total_test_images,total_test_labels,):
    m = np.shape(total_train_images)[1]
    train_images = np.zeros((0, m))
    valid_images = np.zeros((0, m))
    train_labels = np.zeros((0, 10))
    valid_labels = np.zeros((0, 10))
    test_images = np.zeros((0, m))
    test_labels = np.zeros((0, 10))

    for i in range(10):
        onehot = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        onehot[0,i]=1
        tmp_images=total_train_images[np.where(total_train_labels==i)]
        tmp_n=(int)(np.shape(tmp_images)[0]*cf.train_ratio)
        train_images=np.concatenate((train_images,tmp_images[:tmp_n]))
        train_labels=np.concatenate((train_labels,np.matlib.repmat(onehot,np.shape(tmp_images[:tmp_n])[0],1)))
        valid_images=np.concatenate((valid_images,tmp_images[tmp_n:]))
        valid_labels=np.concatenate((valid_labels,np.matlib.repmat(onehot,np.shape(tmp_images[tmp_n:])[0],1)))

        ttmp_images=total_test_images[np.where(total_test_labels==i)]
        test_images=np.concatenate((test_images,ttmp_images))
        test_labels=np.concatenate((test_labels,np.matlib.repmat(onehot,np.shape(ttmp_images)[0],1)))

    return train_images, train_labels,valid_images,valid_labels,test_images,test_labels

"""
Softmax Regression with Regularization
"""
def softmax_regression(total_train_images, total_train_labels,total_test_images,total_test_labels,order=0):

    res=ExperimentResult()
    train_images,train_labels,valid_images,valid_labels,test_images,test_labels=\
        softmax_preprocessing(total_train_images,total_train_labels,total_test_images,total_test_labels)
    # m = np.shape(total_train_images)[1]
    # train_images=np.zeros((0,m))
    # valid_images=np.zeros((0,m))
    # train_labels=np.zeros((0,10))
    # valid_labels=np.zeros((0,10))
    # test_images=np.zeros((0,m))
    # test_labels=np.zeros((0,10))
    #
    # for i in range(10):
    #     onehot = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    #     onehot[0,i]=1
    #     tmp_images=total_train_images[np.where(total_train_labels==i)]
    #     tmp_n=(int)(np.shape(tmp_images)[0]*cf.train_ratio)
    #     train_images=np.concatenate((train_images,tmp_images[:tmp_n]))
    #     train_labels=np.concatenate((train_labels,np.matlib.repmat(onehot,np.shape(tmp_images[:tmp_n])[0],1)))
    #     valid_images=np.concatenate((valid_images,tmp_images[tmp_n:]))
    #     valid_labels=np.concatenate((valid_labels,np.matlib.repmat(onehot,np.shape(tmp_images[tmp_n:])[0],1)))
    #
    #     ttmp_images=total_test_images[np.where(total_test_labels==i)]
    #     test_images=np.concatenate((test_images,ttmp_images))
    #     test_labels=np.concatenate((test_labels,np.matlib.repmat(onehot,np.shape(ttmp_images)[0],1)))
    m = np.shape(total_train_images)[1]
    n = np.shape(train_images)[0]

    eta = cf.sm_eta
    error_cnt = 0
    error_tol = cf.sm_error_tol

    prev_err = 100
    t = 0
    T = cf.sm_T
    w = np.zeros((m,10))
    bestw = np.zeros((m,10))
    best_tr_err=0
    best_va_err=0
    best_te_err=0

    train_y,_,_= softmax_pred(w, train_images, train_labels)

    while error_cnt < error_tol:
        t1 = time.time()
        eta = eta / (1 + t / T)
        term=softmax_regular(w,order)
        w-=term

        delta = np.dot(train_images.T,train_labels-train_y)
        w+=eta*delta/n

        train_y, _, train_err = softmax_pred(w, train_images, train_labels)
        valid_y,_,valid_err=softmax_pred(w,valid_images,valid_labels)
        test_y,_,test_err=softmax_pred(w,test_images,test_labels)

        if valid_err<prev_err:
            error_cnt=0
            bestw=np.array(w)
            best_tr_err=train_err
            best_va_err=valid_err
            best_te_err=test_err
        else:
            error_cnt+=1

        # compute loss-function over entire trainset
        train_loss = softmax_loss(train_y,w,train_images,train_labels,order)
        valid_loss = softmax_loss(valid_y,w, valid_images, valid_labels, order)
        test_loss = softmax_loss(test_y,w, test_images,test_labels, order)
        res.train_err.append(train_err)
        res.valid_err.append(valid_err)
        res.test_err.append(test_err)
        res.train_loss.append(train_loss)
        res.valid_loss.append(valid_loss)
        res.test_loss.append(test_loss)

        t2 = time.time()
        if cf.debug:
            print("Epoch",t,"train_err","{0:.2%}".format(train_err),
                 "valid_err","{0:.2%}".format(valid_err),"test err","{0:.2%}".format(test_err),
                 "cnt",error_cnt,"dt","{0:.4f}".format(t2-t1))
            print("eta", eta, "norm", np.linalg.norm(term), np.linalg.norm(eta * delta / n))
        prev_err = valid_err
        t+=1

    print("Final   train_err", "{0:.2%}  ".format(best_tr_err),
          "valid_err", "{0:.2%}  ".format(best_va_err),
          "test_err", "{0:.2%}".format(best_te_err))
    print()
    for i,wi in enumerate(w.T):
        weightPlot(wi,cf.sm_header+"_"+str(i))

    res.weight_value=bestw
    res.best_train_err=best_tr_err
    res.best_valid_err=best_va_err
    res.best_test_err=best_te_err

    return res


def softmax_pred(w,images, labels):
    n=np.shape(labels)[0]
    preds=np.zeros((n,10))
    #TODO:1. first W^T.X 2. find max minus it 3. exp() 4. generalize()

    score=np.dot(images,w)
    li=np.argmax(score,axis=1)
    for i,j in enumerate(li):
        preds[i,j]=1
    tmp=np.exp(score-np.array([np.max(score,axis=1)]).T)
    score=tmp/np.array([np.sum(tmp,axis=1)]).T
    error_rate = np.sum(preds!=labels)/(2*n)

    # if final:
    #     cnt=0
    #     for i,yec in enumerate(preds):
    #         y = np.where(yec == 1)[0]
    #         t = np.where(labels[i] == 1)[0]
    #         if y==t:
    #             cnt+=1
    #             if cnt%40==0:
    #                 numberPlot(images[i][1:],cf.sm_header[:2]+str(cnt)+"_was"+str(t)+"_but"+str(y))

    return score, preds, error_rate

def softmax_loss(yy,w,images,labels,order):
    n = np.shape(images)[0]
    y=np.array(yy)
    y[yy==0]=cf.very_small_number
    E=np.sum(labels*np.log(y))
    if order == 1:
        return -E / n /10 + cf.sm_lamda * np.sum(np.abs(w))
    elif order == 2:
        return -E / n /10+ cf.sm_lamda * (np.linalg.norm(w) ** 2)
    return -E / n /10

def softmax_regular(w,order):
    term=np.zeros(np.shape(w))
    if order==1:
        term=((term>=0)*2-1)*cf.sm_lamda
    elif order==2:
        term=2*cf.sm_lamda*w
    return term

def numberPlot(x,info="tmp"):
    plt.imshow(np.reshape(x,(28,28)))
    plt.savefig(cf.resdir+"err/"+"num_"+info+".png")
    plt.close()

def weightPlot(w,info="tmp"):
    vis_w = np.reshape(w[1:], (28, 28))
    plt.imshow(vis_w)
    plt.savefig(cf.resdir+"fig_"+info+".png")
    plt.close()

if __name__=="__main__":

    total_train_images, total_train_labels,total_test_images,total_test_labels=load_data()
    total_train_images, total_train_labels = shuffle(total_train_images, total_train_labels)

    # if cf.logistic:
    #     # Great thanks to the following simple case which helps me debug...
    #     # a = np.array([[1, 10, 25], [1, 3, 31], [1, 15, 25], [1, 5, 36], [1, 15, 30], [1, 7, 36]])
    #     # b = np.array([1, 0, 1, 0, 1, 0])
    #     # c = np.array([[1, 18, 25], [1, 8, 21], [1, 7, 38], [1, 3, 32]])
    #     # d = np.array([1, 0, 1, 0])
    #     # logistic_regression(a,b,c,d,c,d,cf.regularization)
    #
    #     res=logistic_regression(train_images,train_labels,test_images,test_labels, cf.lg_reg)
    #
    #     # mine=100
    #     # bestS=0
    #     # for i in [0.1,0.01,0.001,0.0001,0.00001,0.000001]:
    #     #     cf.lg_lamda=i
    #     #     err=logistic_regression(tr_img,tr_lbe,va_img,va_lbe,te_img,te_lbe,cf.lg_reg)
    #     #     if mine>err:
    #     #         mine=err
    #     #         bestS=i
    #     # print(mine,bestS)
    # else:
    #     res=softmax_regression(train_images,train_labels,test_images,test_labels,cf.sm_reg)

    """Experiment Part"""

    # # Test logistic with regularization
    # res=logistic_regression(total_train_images,total_train_labels,total_test_images,total_test_labels, cf.lg_reg)
    # plt.plot(res.train_loss)
    # plt.plot(res.valid_loss)
    # plt.plot(res.test_loss)
    # plt.show()
    #
    # plt.plot(res.train_err)
    # plt.plot(res.valid_err)
    # plt.plot(res.test_err)
    # plt.show()
    #

    # # Test softmax
    # res=softmax_regression(total_train_images,total_train_labels,total_test_images,total_test_labels,cf.sm_reg)
    # plt.plot(res.train_loss)
    # plt.plot(res.valid_loss)
    # plt.plot(res.test_loss)
    # plt.show()
    #
    # plt.plot(res.train_err)
    # plt.plot(res.valid_err)
    # plt.plot(res.test_err)
    # plt.savefig(cf.plotdir + "tmp.png")

    


    print("Finished")