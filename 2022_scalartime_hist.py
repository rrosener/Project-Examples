# Code for analyzing NUMI Scalar data and making histograms, Summer 2022
import ROOT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import pandas as pd
import uproot

# Open ROOT file
mass = '260'
f = mass + 'MeV/dissH_' + mass + '.root'
myFile = ROOT.TFile.Open(f)
myTree = myFile.generator.mevprtl_gen

# Grab data for various branches
m = []
# scalar decay detector coords
X,Y,Z = [],[],[]
# kaon decay detector coords
Xk,Yk,Zk = [],[],[]
# kaon decay beam coords
Xb,Yb,Zb = [],[],[]
E = []
Px, Py, Pz, Pt = [], [], [], []
Px_k, Py_k, Pz_k = [], [], []
gamma = []
t_k = []

for entry in myTree:
    m.append(entry.mevprtl_mom.M())
    X.append(entry.decay_pos.X())
    Y.append(entry.decay_pos.Y())
    Z.append(entry.decay_pos.Z())
    Xk.append(entry.mevprtl_start.X())
    Yk.append(entry.mevprtl_start.Y()) 
    Zk.append(entry.mevprtl_start.Z()) 
    Xb.append(entry.kaon_dpos_beamcoord.X())
    Yb.append(entry.kaon_dpos_beamcoord.Y()) 
    Zb.append(entry.kaon_dpos_beamcoord.Z())
    E.append(entry.mevprtl_mom.Energy())
    Px.append(entry.mevprtl_mom.Px())
    Py.append(entry.mevprtl_mom.Py())
    Pz.append(entry.mevprtl_mom.Pz())
    Pt.append(entry.mevprtl_mom.Pt())
    Px_k.append(entry.kaon_dmom_beamcoord.Px())
    Py_k.append(entry.kaon_dmom_beamcoord.Py())
    Pz_k.append(entry.kaon_dmom_beamcoord.Pz())
    gamma.append(entry.mevprtl_mom.Gamma())
    t_k.append(entry.kaon_dpos_beamcoord.T())

# Convert into arrays
m = np.asarray(m)
m = np.zeros(len(X))
X, Y, Z = np.asarray(X), np.asarray(Y), np.asarray(Z)
Xk, Yk, Zk = np.asarray(Xk), np.asarray(Yk), np.asarray(Zk)
Xb, Yb, Zb = np.asarray(Xb), np.asarray(Yb), np.asarray(Zb)
E = np.asarray(E)
Px, Py, Pz, Pt = np.asarray(Px), np.asarray(Py), np.asarray(Pz), np.asarray(Pt)
Px_k, Py_k, Pz_k = np.asarray(Px_k), np.asarray(Py_k), np.asarray(Pz_k)
gamma = np.asarray(gamma)
t_k = np.asarray(t_k)

# speed of light [cm/s]
c = 3e10
# distance to traveled to ICARUS by scalar [cm]
d_s = np.sqrt((X-Xk)**2 + (Y-Yk)**2 + (Z-Zk)**2)

# scalar velocity from Lorentz factor, momentum, and mass p = γm0v
# total scalar momentum in units of GeV/c
P = np.sqrt((Px)**2 + (Py)**2 + (Pz)**2)
P_k = np.sqrt((Px_k)**2 + (Py_k)**2 + (Pz_k)**2)
# OR v = cp/E, velocity as fraction of c
v_s = c * P / E

# scalar travel time [s]
t_s = d_s / v_s
# convert to [ns]
t_s = t_s * 1e9

# ---------------------------------------------------------
# mini Monte Carlo to calculate proton arrival times

# 84 buckets per batch with 6 batches, last 3 buckets of each batch ignored
buckets = np.arange(1,(84*6)+1)
states = [82,83,0]
mask = np.in1d(buckets%84, states,invert=True)
buckets = buckets[mask]
buckets = 1

# distance between buckets and batches [ns]
bucket_distance = 18.83132779692296
length = 1000000
t_p = np.zeros(length)


for i in list(range(length)):
    # choose random bucket
    bucket = np.random.choice(buckets)
    # bucket's relative position in its batch
    b_pos = bucket%84
   
    # time component from position in beamspill
    t_b = bucket_distance*(bucket-1)
    # time component of bucket modeled as a Gaussian with sigma=0.75ns
    t_g = np.random.normal(0,0.75)
    t_p[i] = t_b + t_g

#------------------------------------------------------------------------

# combined kaon and scalar travel time with proton arrival time
# Subtract t0 via distance from ICARUS to NUMI beam, 799m
# Remove negative t values 
c_i = 3e-1
t = t_k + t_s + t_p
t0 = np.min(t)
t = t - t0

# Histogram weights
# Grab weights using uproot due to bugs with ROOT
file = uproot.open(f)
tree = file['generator/mevprtl_gen']

decay_weight = tree['mevprtl/decay_weight'].array()
flux_weight = tree['mevprtl/flux_weight'].array()
ray_weight = tree['mevprtl/ray_weight'].array()
pot = tree['mevprtl/pot'].array()

# Scale weights by sum of protons-on-target and protons per year
total_pot = np.sum(pot)
scale_pot = 6e20 / total_pot

weights = (decay_weight * flux_weight * ray_weight) * scale_pot 
weights = np.asarray(weights)

# Sort by whether the source kaon decayed-at-rest
kdar = (P_k < 1e-4)
t_kdar = t[kdar]
t_ndar = t[~kdar]
weights_kdar = weights[kdar]
weights_ndar = weights[~kdar]

# Sort by whether the scalar energy is <500MeV
en = (E > 0.5)
t_5m = t[en]
t_n5m = t[~en]
weights_5m = weights[en]
weights_n5m = weights[~en]

# Sort KDAR peaks by whether beamcoord z>720m
peak = (Zb[kdar] >= 72000)
t_kdar_a = t_kdar[peak]
t_kdar_b = t_kdar[~peak]
weights_kdar_a = weights_kdar[peak]
weights_kdar_b = weights_kdar[~peak]



fig, ax = plt.subplots(figsize=(10,10))

# Make plot pretty 
plt.rcParams['figure.dpi']= 300
plt.rc('savefig', dpi=300)
plt.rc('font', size=20)
plt.rc('axes',labelsize=18,titlesize=18)
plt.rc('xtick', direction='out',labelsize=1) 
plt.rc('ytick', direction='out',labelsize=16)
plt.rc('xtick.major', pad=5) 
plt.rc('xtick.minor', pad=5)
plt.rc('ytick.major', pad=5) 
plt.rc('ytick.minor', pad=5)
plt.rc('lines', dotted_pattern = [2., 2.])
plt.rc('figure', titlesize=20)
plt.tick_params(which='major',length=6, width=1.5,labelsize=18)
plt.minorticks_on()
plt.tick_params(which='minor',length=3,width=1)

# Set xticks
lim = 14000
xticks = np.arange(0,lim,100)
ax.set_xticks(xticks)

# Plot histogram
binnum = 25
bins = np.linspace(0,lim,binnum+1)

entries,edges,patch = plt.hist(t_kdar,bins=bins,range=[0,lim],weights=weights_kdar,histtype='step',ec='k',lw=1.75,label='KDAR')
entries2,edges2,patch = plt.hist(t_ndar,bins=bins,range=[0,lim],weights=weights_ndar,histtype='step',ec='r',lw=1.75,label='non-KDAR')

frac = (np.sum(entries[:18]) + (1/28)*entries[19]) / np.sum(entries)
frac2 = (np.sum(entries2[:18]) + (1/28)*entries2[19]) / np.sum(entries2)

# Plot histogram error bars
# Error is sum of weights^2 in each bin
yerr = np.zeros(binnum)
bin_centers = 0.5 * (edges[:-1] + edges[1:])
for i in range(len(edges)-1):
    # Find fraction of events within 10100ns
    if i == 24:
        index = np.where((t_kdar >= edges[i-1]) & (t_kdar < edges[-1]))
    else:
        index = np.where((t_kdar >= edges[i]) & (t_kdar < edges[i+1]))

    w_i = np.sqrt(np.sum(weights_kdar[index]**2))
    yerr[i] = w_i

plt.errorbar(bin_centers, entries, yerr=yerr, fmt='k,',drawstyle='steps',lw=1)

yerr2 = np.zeros(binnum)
bin_centers2 = 0.5 * (edges2[:-1] + edges2[1:])
for i in range(len(edges2)-1):
    if i==24:
        index = np.where((t_ndar >= edges[i-1]) & (t_ndar < edges[-1]))
    else:
        index = np.where((t_ndar >= edges2[i]) & (t_ndar < edges2[i+1]))

    w_i = np.sqrt(np.sum(weights_ndar[index]**2))
    yerr2[i] = w_i

total = (np.sum(entries2[:18]) + (1/28)*entries2[19]) / np.sum(entries)
plt.errorbar(bin_centers2, entries2, yerr=yerr2, fmt='r,',drawstyle='steps',lw=1)

# Plot line at 10100ns
plt.axvline(x=10100,color='k',ls='--',label='NUMI gate')

plt.xlabel('Combined Travel Time (ns)',fontsize=24), plt.ylabel('Relative Normalization',fontsize=24)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
ax.yaxis.offsetText.set_fontsize(24)
plt.title('Combined kaon-scalar travel time with proton arrival time',fontsize=24)

plt.figtext(0.775,0.70,'Scalar mass: ' + mass + 'MeV',fontsize=20)
plt.figtext(0.50,0.40,'KDAR',c='k')
plt.figtext(0.35,0.65,'non-KDAR',c='r')

fractxt = 'Fraction of KDAR\nevents in NUMI gate:\n{frac:.3f}'
fractxt2 = 'Fraction of non-KDAR\nevents in NUMI gate:\n{frac2:.3f}'

plt.figtext(0.775,0.60,fractxt.format(frac=frac),fontsize=20)
plt.figtext(0.775,0.50,fractxt2.format(frac2=frac2),fontsize=20)

plt.legend(loc='upper right',frameon=False,fontsize=16)
plt.show()


