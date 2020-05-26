import os,sys,time
import tensorflow as tf
import numpy as np
import librosa
sys.path.insert(0, './scripts')
sys.path.insert(0, './models')
import e2e_model as nn_model_foreval

# extracting mfcc for input wavfile
FORDER = "./data/dataTrain/test/french/france24/"

# INIT file CSV result
resultCSVfile = open("./result.csv", "w")

def cmvn_slide(feat,winlen=300,cmvn=False): #feat : (length, dim) 2d matrix
# function for Cepstral Mean Variance Nomalization
    maxlen = np.shape(feat)[0]
    new_feat = np.empty_like(feat)
    cur = 1
    leftwin = 0
    rightwin = winlen/2
    
    # middle
    for cur in range(maxlen):
        print(cur-leftwin, int(cur+rightwin))
        cur_slide = feat[cur-leftwin:int(cur+rightwin),:] 
        #cur_slide = feat[cur-winlen/2:cur+winlen/2,:]
        mean =np.mean(cur_slide,axis=0)
        std = np.std(cur_slide,axis=0)
        if cmvn == 'mv':
            new_feat[cur,:] = (feat[cur,:]-mean)/std # for cmvn        
        elif cmvn =='m':
            new_feat[cur,:] = (feat[cur,:]-mean) # for cmn
        if leftwin<winlen/2:
            leftwin+=1
        elif maxlen-cur < winlen/2:
            rightwin-=1    
    return new_feat


def feat_extract(filelist,feat_type,n_fft_length=512,hop=160,vad=True,cmvn=False,exclude_short=500):
# function for feature extracting   
    feat = []
    utt_shape = []
    new_utt_label =[]
    for index,wavname in enumerate(filelist):
        #read audio input
        y, sr = librosa.core.load(wavname,sr=16000,mono=True,dtype='float')

        #extract feature
        if feat_type =='melspec':
            Y = librosa.feature.melspectrogram(y,sr,n_fft=n_fft_length,hop_length=hop,n_mels=40,fmin=133,fmax=6955)
        elif feat_type =='mfcc':
            Y = librosa.feature.mfcc(y,sr,n_fft=n_fft_length,hop_length=hop,n_mfcc=40,fmin=133,fmax=6955)
        elif feat_type =='spec':
            Y = np.abs( librosa.core.stft(y,n_fft=n_fft_length,hop_length=hop,win_length=400) )
        elif feat_type =='logspec':
            Y = np.log( np.abs( librosa.core.stft(y,n_fft=n_fft_length,hop_length=hop,win_length=400) ) )
        elif feat_type =='logmel':
            Y = np.log( librosa.feature.melspectrogram(y,sr,n_fft=n_fft_length,hop_length=hop,n_mels=40,fmin=133,fmax=6955) )

        Y = Y.transpose()
        
        
        # Simple VAD based on the energy
        if vad:
            E = librosa.feature.rms(y, frame_length=n_fft_length,hop_length=hop,)
            threshold= np.mean(E)/2 * 1.04
            vad_segments = np.nonzero(E>threshold)
            if vad_segments[1].size!=0:
                Y = Y[vad_segments[1],:]

                
        #exclude short utterance under "exclude_short" value
        if exclude_short == 0 or (Y.shape[0] > exclude_short):
            if cmvn:
                Y = cmvn_slide(Y,300,cmvn)
            feat.append(Y)
            utt_shape.append(np.array(Y.shape))
#             new_utt_label.append(utt_label[index])
            sys.stdout.write('%s\r' % index)
            sys.stdout.flush()
            
        if index ==0:
            break

        
    tffilename = feat_type+'_fft'+str(n_fft_length)+'_hop'+str(hop)
    if vad:
        tffilename += '_vad'
    if cmvn=='m':
        tffilename += '_cmn'
    elif cmvn =='mv':
        tffilename += '_cmvn'
    if exclude_short >0:
        tffilename += '_exshort'+str(exclude_short)

    return feat, new_utt_label, utt_shape, tffilename #feat : (length, dim) 2d matrix

def analysisDataResult(dialect_index,data):
    weightVN = 0.0
    weightFR = 0.0
    print(dialect_index)
    if(data[3]>0 and data[4]>0 and data[0]<=0 and data[1]<=0 and data[2]<=0 ): weightVN=1.5
    if(data[2]>0 and data[3]>0 and data[4]>0 and data[0]<=0 and data[1]<=0 ): weightFR=1.5
    if(data[0]>0 and data[2]>0 and data[4]>0 and data[1]<=0 and data[3]<=0 ):
        if(dialect_index==4): 
            return ["France",max(data[3],data[4]),min(data[3],data[4])]
        else:
            return ["Vietnamese",min(data[3],data[4]),max(data[3],data[4])]
    if(dialect_index==4 or dialect_index==2): weightFR=1
    if(weightVN > weightFR):
        return ["Vietnamese",min(data[3],data[4]),max(data[3],data[4])]
    return ["France",max(data[3],data[4]),min(data[3],data[4])]

# Feature extraction configuration
FEAT_TYPE = 'logmel'
N_FFT = 400
HOP = 160
VAD = True
CMVN = 'mv'
EXCLUDE_SHORT=0
IS_BATCHNORM = False
IS_TRAINING = False
INPUT_DIM = 40

# Variable Initialization

softmax_num = 5
x = tf.placeholder(tf.float32, [None,None,40])
y = tf.placeholder(tf.int32, [None])
s = tf.placeholder(tf.int32, [None,2])

emnet_validation = nn_model_foreval.nn(x,y,y,s,softmax_num,IS_TRAINING,INPUT_DIM,IS_BATCHNORM);
sess = tf.InteractiveSession()
saver = tf.train.Saver()
tf.initialize_all_variables().run()

#get file
LISTFILE = [FORDER+f for f in os.listdir(FORDER) if os.path.isfile(os.path.join(FORDER, f))]
RESULT =[]

resultCSVfile.write("{}, {}\n".format("FILE NAME", "LANGUAGE IDENTIFICATION"))
for FILE in LISTFILE:
    FILENAME = [FILE]
    start_time = time.time()
    feat, utt_label, utt_shape, tffilename = feat_extract(FILENAME,FEAT_TYPE,N_FFT,HOP,VAD,CMVN,EXCLUDE_SHORT)
    elapsed_time = time.time() - start_time
    ### Loading neural network 
    saver.restore(sess,'./data/pretrained_model/model1284000.ckpt-1284000')
    start_time = time.time()
    likelihood= emnet_validation.o1.eval({x:feat, s:utt_shape})
    elapsed_time = time.time() - start_time
    dialect_index = np.argmax(likelihood)
    
    dataBeforeDetect = analysisDataResult(int(dialect_index),likelihood[0])
    FILENAME[0]= FILENAME[0].split("/")[-1]
    # RESULT.append(FILENAME[0] +" ====> "+ dataBeforeDetect)
    resultCSVfile.write("{}, {}, {}, {}\n".format(FILENAME[0], dataBeforeDetect[0], dataBeforeDetect[1], dataBeforeDetect[2]))
#clear screen and print result
clear = lambda: os.system('clear') #on Linux System
clear()
print("RESULTS")
for i in RESULT:
    print(i)
print("------------------------------")
print("RESULT SAVE TO FILE result.csv")
