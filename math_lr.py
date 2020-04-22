import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, lfilter_zi, welch
import random
from scipy.stats import pearsonr

def readFile(fileName):
    data = [[], []]
    with open(fileName, 'r') as f:
        for row in f:
            row=row[7:]
            rowList = row.split("  ")
            if len(rowList)!=2:
                rowList = row.split(" ")
            data[0].append(rowList[0][:-1])
            if len(rowList)==2:
                data[1].append(float(rowList[1][:-1]))
    return np.asarray(data[1]), np.asarray(data[0])


def drawSignal(time, signal, fmin, fmax):
    plt.figure(figsize=(13,8))
    plt.plot(time,signal)
    plt.xlabel(u'time')
    plt.ylabel(u'signal')
    plt.title(u'ECG')
    plt.grid()
    plt.show()
    fourier = np.fft.fft(signal)
    T = time[-1]-time[0]
    td = T/len(signal)
    freqs = (time-time[0])/(T*td)
    L = len(signal)//2
    plt.figure(figsize=(13,8))
    plt.plot(freqs[0:L], np.abs(fourier)[0:L])
    plt.xlabel(u'frequency')
    plt.ylabel(u'spectrum')
    plt.grid()
    plt.title(u'Spectral density')
    if (fmin>-1):
        plt.plot([    fmax/T,       fmax/T, fmin/T,       fmin/T], 
         [np.abs(fourier)[fmax],   0,    0,    np.abs(fourier)[fmin]])
    plt.show()
    return
    
    
def interp_signal(time, signal):
    T = time[-1]-time[0]
    td = T/len(signal)
    interpolated = interp1d(time, signal, kind='cubic')
    timeNew = np.linspace(0,T,10*len(signal))
    freqsNew = timeNew/(T*0.1*td)
    interpol = interpolated(timeNew)
    # print(np.shape(interpol))
    drawSignal(timeNew,interpol,-1,-1)
    return timeNew, interpol
    
    
def PhaseMod(time, signal, w0):
    T = time[-1]-time[0]
    td = T/len(signal)
    PM = []
    length = len(signal)
    Cumsum = np.cumsum(signal)
    Cumsum = Cumsum / np.amax(Cumsum)
    for i in range(length):
        phase = w0*i*td+3.14*Cumsum[i]
        PM.append(np.sin(phase))
        
    return np.asarray(PM)


def findBorders(spectrum, a):
    L = len(spectrum) // 2
    spectrum = spectrum[0:L]
    sdens = np.abs(spectrum) ** 2
    integr = np.cumsum(sdens)
    integr = normalize_signal(integr,1)
    fmin = np.where(integr < a)[0][-1]
    fmax = np.where(integr > 1-a)[0][0]
    return fmin, fmax


def normalize_signal(signal, crit):
    Max = np.max(signal)
    Min = np.min(signal)
    Mean = np.mean(signal)
    if crit>0.5:
        if Max>Min:
            return (signal-Min)/(Max-Min)
        else:
            return []
    else:
        if Max>Min:
            return (signal-Mean)/(Max-Min)
        else:
            return []
    
    
def rect(length, x0, t, A):
    rec = np.zeros(length)
    rec[x0: x0+t] = A
    return rec


def rect1(line):
    rec = np.zeros(line.size)
    for i in range(len(line)):
        if np.abs(line[i]) < 0.5:
            rec[i] = 1
            
    return rec


def TIM(signal, dw):
    length = len(signal)
    tim = np.zeros(length)
    # Cumsum = np.cumsum(signal)
    i = 0
    while i<length:
        F = signal[i] + 1
        T = np.int32(dw // F)
        # print(T)
        A = 1
        x0 = i
        tim += rect(length, x0, T // 2, A)
        i += T
        
    return tim


def FIM(signal, dw, T, t):
    length = len(signal)
    time = np.linspace(0,length,length)
    integr = np.cumsum(signal)
    MAX = np.max(integr)
    I = length/T + dw*MAX/6.283
    I = np.int32(I)
    fim = np.zeros(signal.size)
    for i in range(I):
        fim += rect1((time - i*T + integr*dw*T/6.283)/t)
        
    return fim


def AM(time,signal,w0):
    am = signal
    T = time[-1]-time[0]
    td = T/len(signal)
    for i in range(len(signal)):
        am[i] *= np.cos(w0*td*i)
        
    return am


def butter_lowpass(cut, fs, order=3):
    nyq = 0.5 * fs
    cut = cut / nyq
    b, a = butter(order, cut, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, cut, fs, order=3):
    b, a = butter_lowpass(cut, fs, order=order)
    zi = lfilter_zi(b, a)
    y, z = lfilter(b, a, data, zi=zi*data[0])
    return y


def deAM(time,am,w0):
    deam = am
    T = time[-1]-time[0]
    td = T/len(am)
    fs = 1/td
    for i in range(len(am)):
        deam[i] *= np.cos(w0*td*i)
    deam = butter_lowpass_filter(deam, w0/(2*3.1415926), fs)
    return deam


def add_noise(signal):
    signal = normalize_signal(signal,0)
    N = len(signal)
    a = 2000
    for i in range(N):
        signal[i] += random.random()/a - 1/(2*a)
        
    return signal


def correlation(sig1,sig2):
    fourier1 = np.abs(np.fft.fft(sig1))
    fourier2 = np.abs(np.fft.fft(sig2))
    c = pearsonr(fourier1, fourier2)
    return c[0]


def get_SNR(signal0, signal1, fmin, fmax):
    fourier0 = np.fft.fft(signal0)[fmin:fmax]
    fourier1 = (np.fft.fft(signal1)[fmin:fmax])
    f0 = np.abs(fourier0) ** 2
    f1 = np.abs(fourier1) ** 2
    P0 = np.sum(f0)
    P1 = np.sum(f1)
    SNR = P0/P1
    return SNR


def stats(time, signal, fmin, fmax):
    N = len(signal) // 20
    w = np.linspace(25,N*6.283,1000)
    SNR = []
    CORR = []
    for w0 in np.ndarray.tolist(w):
        am = AM(time, signal, w0)
        amn = add_noise(am)
        noise = deAM(time, amn-am, w0)
        deam = deAM(time, amn, w0)
        SNR.append(get_SNR(signal,noise,fmin,fmax))
        CORR.append(correlation(deam, signal))

    plt.figure(figsize=(13,8))
    plt.plot(w/6.283, SNR)
    plt.xlabel(u'frequency')
    plt.ylabel(u'Ratio')
    plt.title(u'SNR')
    plt.grid()
    plt.show()

    plt.figure(figsize=(13,8))
    plt.plot(w/6.283, np.abs(CORR))
    plt.xlabel(u'frequency')
    plt.ylabel(u'Corr')
    plt.title(u'Correlation')
    plt.grid()
    plt.show()
    
    return