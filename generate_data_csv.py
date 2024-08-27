import sys, os

datadir = '/shared/u/v_sinuo_liu/class_1_100/SNR005'
classes = {}
for line in open('classname_label.txt').readlines():
    name, cid = line.rstrip().split(',')
    classes[name] = cid

fpout = open('data_split/simulated_SNR005_train.csv', 'w')
fpout.write('path,label\n')

for name, cid in classes.items():
    for i in range(0, 500):
        line = '%s/%s/subtomogram_png/tomotarget%d.png,%s\n' % (datadir, name, i, cid)
        fpout.write(line)
fpout.close() 
