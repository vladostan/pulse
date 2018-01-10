import json
from scipy.signal import butter, filtfilt
import pandas as pd
from pandas import DataFrame, Series
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix
from scipy.interpolate import interp1d

from preprocessing.preprocessing import PreProcessor
from preprocessing.ssa import SingularSpectrumAnalysis
import scipy.linalg as lin

matplotlib.rc('xtick', labelsize=10)
matplotlib.rc('ytick', labelsize=10)
matplotlib.rc('axes', titlesize=15)
matplotlib.rc('legend', fontsize=15)
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('channels_reconstructed.pdf')

index=0
window_size=16

fsamp = 250
tsample = 1 / fsamp
f_low = 50
f_high = 1
order = 2


project_file_path = "/home/runge/openbci/git/OpenBCI_Python"
config_file = project_file_path + "/config/config.json"
raw_reconstructed_signals = pd.read_csv(project_file_path+"/build/dataset/train/result/bycept_reconstructed_signals.csv")
# raw_channels_data = pd.read_csv(project_file_path+"/build/dataset/result_bicep.csv").ix[:,2:7].dropna()
# scatter_matrix(raw_channels_data, alpha=0.2, figsize=(6, 6), diagonal='kde')
channels_names = ["ch1", "ch2", "ch3", "ch4", "ch5"]


# channel_vector = [1,2, 3, 4, 5]
# n_ch = len(channel_vector)
# df = pd.read_csv("/home/runge/openbci/application.linux64/application.linux64/OpenBCI-RAW-right_strait_up_new.txt")
# df = df[channel_vector].dropna(axis=0)
#
# processed_signal_channel = df.copy()
#
# b, a = butter(order, (order * f_low * 1.0) / fsamp * 1.0, btype='low')
# for i in range(0, n_ch):
#     processed_signal_channel.ix[:, i] = np.transpose(filtfilt(b, a, df.ix[:, i]))
#
# b1, a1 = butter(order, (order * f_high * 1.0) / fsamp * 1.0, btype='high')
# for i in range(0, n_ch):
#     processed_signal_channel.ix[:, i] = np.transpose(filtfilt(b1, a1, processed_signal_channel.ix[:, i]))
#
# Wn = (np.array([58.0, 62.0]) / 500 * order).tolist()
# b3, a3 = butter(order, Wn, btype='stop')
# for i in range(0, n_ch):
#     processed_signal_channel.ix[:, i] = np.transpose(filtfilt(b3, a3, processed_signal_channel.ix[:, i]))



fsamp = 1

graph_legend = []
handle_as=[]
labels_as=[]

num_ch = len(channels_names)

start = 2100
end = 2600

raw_processed_signal = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/build/dataset/train/result/reconstructed_bycept_kinect__angles_.csv").dropna()

# scatter_matrix(processed_signal, alpha=0.2, figsize=(6, 6), diagonal='kde')

def nomalize_signal(input_signal):
    mean = np.mean(input_signal, axis=0)
    input_signal -= mean
    return input_signal / np.std(input_signal, axis=0)

start_g = 700
end_g = 10000
processed_signal = raw_processed_signal.copy()
nomalized_signal = nomalize_signal(processed_signal)
# mapping = interp1d([-1,1],[0,180])
pattern=np.array(nomalized_signal.ix[:, 1][start :end])
data=np.array(nomalized_signal.ix[:, 1][start_g :end_g])

def create_mats(dat):
    step=5
    eps=.1
    dat=dat[::step]
    K=len(dat)+1
    A=np.zeros((K,K))
    A[0,1]=1.
    pA=np.zeros((K,K))
    pA[0,1]=1.
    for i in xrange(1,K-1):
        A[i,i]=(step-1.+eps)/(step+2*eps)
        A[i,i+1]=(1.+eps)/(step+2*eps)
        pA[i,i]=1.
        pA[i,i+1]=1.
    A[-1,-1]=(step-1.+eps)/(step+2*eps)
    A[-1,1]=(1.+eps)/(step+2*eps)
    pA[-1,-1]=1.
    pA[-1,1]=1.

    w=np.ones( (K,2) , dtype=np.float)
    w[0,1]=dat[0]
    w[1:-1,1]=(dat[:-1]-dat[1:])/step
    w[-1,1]=(dat[0]-dat[-1])/step

    return A,pA,w,K

A,pA,w,K=create_mats(pattern)

eta=5. #precision parameter for the autoregressive portion of the model
lam=.1 #precision parameter for the weights prior

N=1 #number of sequences
M=2 #number of dimensions - the second variable is for the bias term
T=len(data) #length of sequences

x=np.ones( (T+1,M) ) # sequence data (just one sequence)
x[0,1]=1
x[1:,0]=data

#emissions
e=np.zeros( (T,K) )
#residuals
v=np.zeros( (T,K) )

#store the forward and backward recurrences
f=np.zeros( (T+1,K) )
fls=np.zeros( (T+1) )
f[0,0]=1
b=np.zeros( (T+1,K) )
bls=np.zeros( (T+1) )
b[-1,1:]=1./(K-1)

#hidden states
z=np.zeros( (T+1),dtype=np.int )

#expected hidden states
ex_k=np.zeros((T,K))

# expected pairs of hidden states
ex_kk=np.zeros((K,K))
nkk=np.zeros((K,K))

def fwd():
    global f,e
    for t in xrange(T):
        f[t+1,:]=np.dot(f[t,:],A)*e[t,:]
        sm=np.sum(f[t+1,:])
        fls[t+1]=fls[t]+np.log(sm)
        f[t+1,:]/=sm
        assert f[t+1,0]==0

def bck():
    global b,e
    for t in xrange(T-1,-1,-1):
        b[t,:]=np.dot(A,b[t+1,:]*e[t,:])
        sm=np.sum(b[t,:])
        bls[t]=bls[t+1]+np.log(sm)
        b[t,:]/=sm

def em_step(xn):
    global A,w,eta
    global f,b,e,v
    global ex_k,ex_kk,nkk

    x=xn[:-1] #current data vectors
    y=xn[1:,:1] #next data vectors predicted from current
    #compute residuals
    v=np.dot(x,w.T) # (N,K) <- (N,1) (N,K)
    v-=y
    e=np.exp(-eta/2*v**2,e)

    fwd()
    bck()

    # compute expected hidden states
    for t in xrange(len(e)):
        ex_k[t,:]=f[t+1,:]*b[t+1,:]
        ex_k[t,:]/=np.sum(ex_k[t,:])

    # compute expected pairs of hidden states
    for t in xrange(len(f)-1) :
        ex_kk=A*f[t,:][:,np.newaxis]*e[t,:]*b[t+1,:]
        ex_kk/=np.sum(ex_kk)
        nkk+=ex_kk

    # max w/ respect to transition probabilities
    A=pA+nkk
    A/=np.sum(A,1)[:,np.newaxis]

    # solve the weighted regression problem for emissions weights
    #  x and y are from above
    for k in xrange(K):
        ex=ex_k[:,k][:,np.newaxis]
        dx=np.dot(x.T,ex*x)
        dy=np.dot(x.T,ex*y)
        dy.shape=(2)
        w[k,:]=lin.solve(dx+lam*np.eye(x.shape[1]), dy)

    #return the probability of the sequence (computed by the forward algorithm)
    return fls[-1]

for i in xrange(5):
    print em_step(x)

#get rough boundaries by taking the maximum expected hidden state for each position
r=np.arange(len(ex_k))[np.argmax(ex_k,1)<2]
f = np.diff(np.diff(r))
for i in range(0,len(f)):
    if(f[i]<=0):
        r[i] = 0
#plot
plt.plot(range(T),x[1:,0])
yr=[np.min(x[:,0]),np.max(x[:,0])]
for i in r:
        plt.plot([i,i],yr,'-r')
plt.show()


graph_legend = []
fig = plt.figure(figsize=(20, 14))
fig.subplots_adjust(hspace=.5)
x = np.arange(start_g,end_g, 1)
for i in range(0,5):
    ax = plt.subplot(810+i+1)
    # axes = plt.gca()
    l1=ax.plot(range(T),raw_reconstructed_signals.ix[:,i][start_g:end_g], linewidth=1.0,
               label="Processed signal with SSA")
    graph_legend.append(l1)
    handles, labels = ax.get_legend_handles_labels()
    handle_as.append(handles)
    labels_as.append(labels)
    plt.title(channels_names[i])
    for i in r:
            plt.plot([i,i],yr,'-r')

for j in range(0,3):
    ax = plt.subplot(815+1+j)
        # axes = plt.gca()
    l1=ax.plot(range(T),raw_processed_signal.ix[:,j][start_g:end_g], linewidth=1.0,
               label="Processed signal with SSA")
    graph_legend.append(l1)
    handles, labels = ax.get_legend_handles_labels()
    handle_as.append(handles)
    labels_as.append(labels)
    plt.title(channels_names[j])
    for i in r:
            plt.plot([i,i],yr,'-r')

fig.legend(handles=handle_as[0], labels=labels_as[0])
fig.text(0.5, 0.04, 'position', ha='center', fontsize=10)
fig.text(0.04, 0.5, 'angle(0-180)', va='center', rotation='vertical', fontsize=10)
plt.show()


# fig = plt.figure(figsize=(20, 14))
# fig.subplots_adjust(hspace=.5)
#
# start_g = 0
# end_g= 200
#
# for i in range(0, 5):
#     x = np.arange(start_g, end_g, 1)
#     input_signal = raw_reconstructed_signals.ix[:,i][start_g:end_g]




# with open(config_file) as config:
#     config = json.load(config)
#     config["train_dir_abs_location"] = project_file_path + "/build/dataset/train"
#
#     for h in range(0, num_ch):
#         preprocessor = PreProcessor(h, None, None, config)
#         ax = plt.subplot(num_ch,1,h+1)
#
#         axes = plt.gca()` 
#         # axes.set_ylim([0, 180])
#         if(end==0):
#             end = raw_channels_data.ix[:, h].shape[0]-1
#         x = np.arange(start, end, 1)
#         input_signal = raw_channels_data.ix[:, h][start * fsamp:end * fsamp]
#         # l1 = ax.plot(input_signal, linewidth=3.0, label="raw signal")
#         # graph_legend.append(l1)
#
#         noise_reducer_signal = preprocessor.apply_noise_reducer_filer(input_signal)
#         l2 = ax.plot(x, noise_reducer_signal, linewidth=3.0, label="noise_reducer_signal")
#         graph_legend.append(l2)
#
#         # normalize_signal = preprocessor.nomalize_signal(noise_reducer_signal)
#         # l3 = ax.plot(x, normalize_signal, linewidth=3.0, label="normalize_signal")
#         # graph_legend.append(l3)
#
#         reconstructed_signal = SingularSpectrumAnalysis(noise_reducer_signal, window_size, False).execute(1)
#         l4 = ax.plot(x,reconstructed_signal, linewidth=3.0, label='reconstructed signal with SSA')
#         graph_legend.append(l4)
#
#         handles, labels = ax.get_legend_handles_labels()
#         handle_as.append(handles)
#         labels_as.append(labels)
#         plt.title(channels_names[h])
#         # leg = plt.legend(handles=handles, labels=labels)
#
#     fig.legend(handles=handle_as[0], labels=labels_as[0])
#     fig.text(0.5, 0.04, 'position', ha='center', fontsize=10)
#     fig.text(0.04, 0.5, 'angle(0-180)', va='center', rotation='vertical', fontsize=10)
#     plt.show()


# pp.savefig(bbox_inches='tight')
# pp.close()


# def plot_detected_pattern(self, start=0, end=0, is_raw=False, pattern_start_with=200, pattern_end_with=800):
#     if is_raw:
#         channel_signals = pd.read_csv(self.dataset_location).ix[:, 2:7].dropna()
#     else:
#         channel_signals = pd.read_csv(self.config["train_dir_abs_location"]
#                                       + "/result/raw_reconstructed_signalbycepts.csv").dropna()
#     kinect_angles = pd.read_csv(self.config["train_dir_abs_location"]
#                                 + "/result/reconstructed_bycept_kinect__angles_.csv").dropna()
#     nomalized_signal = self.nomalize_signal(kinect_angles)
#     pattern = np.array(nomalized_signal.ix[:, 1][pattern_start_with:pattern_end_with])
#
#     if end == 0:
#         end = len(nomalized_signal)
#
#     data = np.array(nomalized_signal.ix[:, 1][start:end])
#     graph_legend = []
#     handle_as = []
#     labels_as = []
#
#     def create_mats(dat):
#         step = 5
#         eps = .1
#         dat = dat[::step]
#         K = len(dat) + 1
#         A = np.zeros((K, K))
#         A[0, 1] = 1.
#         pA = np.zeros((K, K))
#         pA[0, 1] = 1.
#         for i in xrange(1, K - 1):
#             A[i, i] = (step - 1. + eps) / (step + 2 * eps)
#             A[i, i + 1] = (1. + eps) / (step + 2 * eps)
#             pA[i, i] = 1.
#             pA[i, i + 1] = 1.
#         A[-1, -1] = (step - 1. + eps) / (step + 2 * eps)
#         A[-1, 1] = (1. + eps) / (step + 2 * eps)
#         pA[-1, -1] = 1.
#         pA[-1, 1] = 1.
#
#         w = np.ones((K, 2), dtype=np.float)
#         w[0, 1] = dat[0]
#         w[1:-1, 1] = (dat[:-1] - dat[1:]) / step
#         w[-1, 1] = (dat[0] - dat[-1]) / step
#         return A, pA, w, K
#
#     self.A, pA, w, K = create_mats(pattern)
#
#     eta = 5.
#     lam = .1
#     N = 1
#     M = 2
#     T = len(data)
#     x = np.ones((T + 1, M))
#     x[0, 1] = 1
#     x[1:, 0] = data
#
#     # emissions
#     e = np.zeros((T, K))
#     # residuals
#     v = np.zeros((T, K))
#
#     # store the forward and backward recurrences
#     f = np.zeros((T + 1, K))
#     fls = np.zeros((T + 1))
#     f[0, 0] = 1
#     b = np.zeros((T + 1, K))
#     bls = np.zeros((T + 1))
#     b[-1, 1:] = 1. / (K - 1)
#
#     # hidden states
#     z = np.zeros((T + 1), dtype=np.int)
#
#     # expected hidden states
#     ex_k = np.zeros((T, K))
#
#     # expected pairs of hidden states
#     ex_kk = np.zeros((K, K))
#     nkk = np.zeros((K, K))
#
#     def fwd():
#         global f, e
#         for t in xrange(T):
#             f[t + 1, :] = np.dot(f[t, :], self.A) * e[t, :]
#             sm = np.sum(f[t + 1, :])
#             fls[t + 1] = fls[t] + np.log(sm)
#             f[t + 1, :] /= sm
#             assert f[t + 1, 0] == 0
#
#     def bck():
#         global b, e
#         for t in xrange(T - 1, -1, -1):
#             b[t, :] = np.dot(self.A, b[t + 1, :] * e[t, :])
#             sm = np.sum(b[t, :])
#             bls[t] = bls[t + 1] + np.log(sm)
#             b[t, :] /= sm
#
#     def em_step(xn):
#         global self.A, w, eta
#         global f, b, e, v
#         global ex_k, ex_kk, nkk
#
#         x = xn[:-1]  # current data vectors
#         y = xn[1:, :1]  # next data vectors predicted from current
#         # compute residuals
#         v = np.dot(x, w.T)  # (N,K) <- (N,1) (N,K)
#         v -= y
#         e = np.exp(-eta / 2 * v ** 2, e)
#
#         fwd()
#         bck()
#
#         # compute expected hidden states
#         for t in xrange(len(e)):
#             ex_k[t, :] = f[t + 1, :] * b[t + 1, :]
#             ex_k[t, :] /= np.sum(ex_k[t, :])
#
#         # compute expected pairs of hidden states
#         for t in xrange(len(f) - 1):
#             ex_kk = self.A * f[t, :][:, np.newaxis] * e[t, :] * b[t + 1, :]
#             ex_kk /= np.sum(ex_kk)
#             nkk += ex_kk
#
#         # max w/ respect to transition probabilities
#         self.A = pA + nkk
#         self.A /= np.sum(A, 1)[:, np.newaxis]
#
#         # solve the weighted regression problem for emissions weights
#         #  x and y are from above
#         for k in xrange(K):
#             ex = ex_k[:, k][:, np.newaxis]
#             dx = np.dot(x.T, ex * x)
#             dy = np.dot(x.T, ex * y)
#             dy.shape = (2)
#             w[k, :] = lin.solve(dx + lam * np.eye(x.shape[1]), dy)
#
#         # return the probability of the sequence (computed by the forward algorithm)
#         return fls[-1]
#
#     for i in xrange(5):
#         em_step(x)
#
#     # get rough boundaries by taking the maximum expected hidden state for each position
#     r = np.arange(len(ex_k))[np.argmax(ex_k, 1) < 2]
#     f = np.diff(np.diff(r))
#     for i in range(0, len(f)):
#         if (f[i] <= 0):
#             r[i] = 0
#     # plot
#     plt.plot(range(T), x[1:, 0])
#     yr = [np.min(x[:, 0]), np.max(x[:, 0])]
#     for i in r:
#         plt.plot([i, i], yr, '-r')
#     plt.show()
#
#     graph_legend = []
#     fig = plt.figure(figsize=(20, 14))
#     fig.subplots_adjust(hspace=.5)
#     x = np.arange(start, end, 1)
#     for i in range(0, 5):
#         ax = plt.subplot(810 + i + 1)
#         # axes = plt.gca()
#         l1 = ax.plot(range(T), channel_signals.ix[:, i][start:end], linewidth=1.0,
#                      label="Processed signal with SSA")
#         graph_legend.append(l1)
#         handles, labels = ax.get_legend_handles_labels()
#         handle_as.append(handles)
#         labels_as.append(labels)
#         plt.title(self.channels_names[i])
#         for i in r:
#             plt.plot([i, i], yr, '-r')
#
#     for j in range(0, 3):
#         ax = plt.subplot(815 + 1 + j)
#         l1 = ax.plot(range(T), kinect_angles.ix[:, j][start:end], linewidth=1.0,
#                      label="Processed signal with SSA")
#         graph_legend.append(l1)
#         handles, labels = ax.get_legend_handles_labels()
#         handle_as.append(handles)
#         labels_as.append(labels)
#         plt.title(self.channels_names[j])
#         for i in r:
#             plt.plot([i, i], yr, '-r')
#
#     fig.legend(handles=handle_as[0], labels=labels_as[0])
#     fig.text(0.5, 0.04, 'position', ha='center', fontsize=10)
#     fig.text(0.04, 0.5, 'angle(0-180)', va='center', rotation='vertical', fontsize=10)
#     plt.show()
