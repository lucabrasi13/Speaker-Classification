import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.fftpack import dct
from scipy.signal import find_peaks
from scipy.io import savemat
import warnings
import os,glob

## ----------------- Function declaration --------------------------------##
def freq_to_mel(f):
	""" Convert from freq to mel scale
	
	:param f: freq in Hz
	:return:
	"""
	return 1125*np.log(1+(f/700))

def mel_to_freq(m):
	""" Convert from MEL to freq

	:param m: Frequency in Mel scale
	:return:
	"""
	return 700*(np.exp((m/1125))-1)

def round_nearest_fft_bin(h):
	""" Find the nearest FFT bin for a N-point FFT for a given freq in Hz
	
	:param h:Frequency in Hz
	:return: FFT bin number
	""" 
	return np.floor((NFFT+1)*h/(2*FS))

def compute_triangle(p1,p2,p3):
	""" Triangle starting at p1, peaking at p2 and decreasing to zero at p3

	:param p1,p2,p3: Points to compute the geometry of the triangle
	:return:
	"""
	triag = np.zeros(int(NFFT))
	for k in range(0,int(NFFT)):
		if(k<p1):
			triag[k] = 0
		elif(p1<=k<=p2):
			triag[k] = (k-p1)/(p2-p1)
		elif(p2<=k<=p3):
			triag[k] = (p3-k)/(p3-p2)
		elif(k>p3):
			triag[k] = 0
		else:
			continue
	return triag

def generate_MFCC(flag):
	""" Generate the MFCC coefficients
	
	:param: flag to plot the visual representation of the filterbank
	:return: Return the matrix contining all NB_FBANK	
	"""
	H = np.zeros((NB_FBANK,NFFT))
	for i in range(1,len(F_NB)-1):
		H[i-1,:] = compute_triangle(F_NB[i-1],F_NB[i],F_NB[i+1])
	if(flag == 1):
		overlay_plot(H)
		return H
	else:	
		return H	
		
def overlay_plot(H):
	""" Support function for generate_MFCC
	:param: MFCC coefficients
	:return: Plot of the filterbanks
	"""
	for i in range(0,H.shape[0]):
		plt.plot(H[i,:])
	plt.title("MFCC filterbanks overlayed")
	plt.xlabel("DFT bins -->")
	plt.ylabel("H(f) -->")
	plt.show()

def compute_energy(s):
	""" Function to compute the energy of each frame
	
	:param s: Speech signal
	:return: Return the Energy measure of each frame	
	"""
	E = 0.
	for i in range(0,len(s)):
		E = E+s[i]**2	
	return E

def buffer_signal(s,Twin,p):
	""" Function similar to MATLAB's buffer functions

	:param s: Speech signal of length N
	:param Twin: Temporal length of the signal in ms
	:param p: Number of samples to consider the overlap(p<L)	
	:return: The windowed speech signal with overlap of p samples
	"""
	L = int(Twin*FS)
	window = np.hamming(L)
	N = len(s)-np.mod(len(s),L)
	M = int(np.floor(N/L))
	x = np.zeros(N)
	for i in range(0,int(M-1)):
		if(i==0):
			x[0:L] = np.multiply(s[0:L],window); 
		else:
			x[(i*L)-p:((i+1)*L)-p] = np.multiply(s[(i*L)-p:((i+1)*L)-p],window)
	return np.reshape(x,(M,L)).T

def gen_MFCC_features(s,Twin,p,H):
	""" Function to generate the MFCC features
	
	:param: s: Speech signal
	:param: Twin: Window length in ms
	:param:	p: Number of samples to consider the overlap(p<L)
	:param: H: MFCC filterbank	
	:return:
	"""
	sw = buffer_signal(s,Twin,p)
	#sw = vus_detection(s,Twin,p)
	W = np.array(np.abs(np.fft.fft(sw,NFFT,axis=0)))
	EF = dct(np.array(H*W)).T
	for i in range(0,EF.shape[0]):
		EF[i,0] = compute_energy(sw[:,i])
	return EF

def zcr(s):
	""" Function to find the zero crossing rate of a speech frame
	
	:param: s:speech frame
	:return:	
	"""

	return len(np.where(np.diff(np.sign(s)))[0])

def short_time_energy(s):
	""" Function to find the short time energy of a speech frame
	
	:param: s:speech frame
	:return:	
	
	"""
	E = 0
	for i in range(0,len(s)):
		E = E + np.power(s[i],2)
	return E

def vus_detection(s,Twin,p):
	""" Function to find the frames in the voiced region

	:param: s: speech frame
	:param: Twin
	:param: p
	:return: Return buffered matrix without unvoiced frames
	
	"""
	s = s/np.max(s)
	sw = buffer_signal(s,Twin,p)
	M = sw.shape[0]
	ZCR = np.zeros(M)
	E = np.zeros(M)
	Ix = []
	for i in range(0,M):
		ZCR[i] = zcr(sw[i,:])
		E[i] = short_time_energy(sw[i,:])
		if(not((ZCR[i]<ZCR_PARAM) and (E[i]>STE_PARAM))):
			Ix = np.append(Ix,i)			
	return np.delete(sw,Ix,0)

def find_frame_pitch(s):
	""" Return the pitch from a particular frame

	:param: s
	:param: FL
	:param: FU
	:return:	
	
	"""
	ms1 = int(FS/FU)
	ms2 = int(FS/FL)
	C = np.fft.ifft(np.log(np.abs(np.fft.fft(s))))
	Ix = np.argmax(np.abs(C[ms1:ms2]))
	return FS/(ms1+Ix-1)
	
def gen_pitch_estimate(s,Twin,p):
	""" Return the pitch of the entire speech signal

	:param sw
	:param FU: Upper passband freq for liftering
	:param FL: Lower passband freq for liftering
	:return:

	"""
	sw = buffer_signal(s,Twin,p)
	L = sw.shape[1]
	p = np.zeros(L)
	for i in range(0,L):	
		p[i] = find_frame_pitch(sw[i,:])	
	return p

## ---------------------------------------------------------------------##

## Ignore the Deprecation Warning
warnings.filterwarnings("ignore", category=DeprecationWarning)	

## Global variable declaration
global FS
global FL 
global FH
global NB_FBANK
global NFFT
global NB_MFCC
global ZCR_PARAM
global STE_PARAM
global FU
global FL

## Global variable definition
FS = 8e3
FL = 3e2
FH = FS
NB_FBANK = 40
NFFT = 512
NB_MFCC = 13
ZCR_PARAM = 13
STE_PARAM = 0.001
FU = 1000
FL = 50

## Gen. NB_FBANK+2 points
[ML,MH] = map(freq_to_mel,[FL,FH])
M = np.linspace(ML,MH,NB_FBANK+2)

## Convert back to freq
F = np.array(list(map(mel_to_freq,M)))

## Round to the nearest FFT bin
F_NB = np.array(list(map(round_nearest_fft_bin,F)))

## Generate the filterbanks
H = np.matrix(generate_MFCC(0))

## Extract the MFCC and Pitch features
a = []
EF_MFCC = []
EF_P = []
Twin = 30e-3
p = int(0.75*(Twin*FS))
path = '/home/pheonix13/Documents/Workspace/Python/Machine Learning/Speaker Classification/Git'
for filename in glob.glob(os.path.join(path, '*.wav')):
	_,a = read(filename)
	a = a/np.max(a)
	list.append(EF_MFCC,np.matrix(gen_MFCC_features(a,Twin,p,H)))
	list.append(EF_P,np.array(gen_pitch_estimate(a,Twin,p)))

feature = {}
feature['MFCC_features'] = EF_MFCC
feature['MFCC_pitch'] = EF_P
savemat('MFCC_features.mat',feature)

