import random

n_shot = 3
filename = "complete_real_split.csv"
fp_csv = open(filename,'r')
shuffle_data = fp_csv.readlines()[1:]
random.shuffle(shuffle_data)
labels = {"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0}
train_outputs = {}
test_outputs = {}
for i,line in  enumerate(shuffle_data):
    path,label = line.split(",")
    label = label[0]
    if (labels[label] < n_shot):
        train_outputs[path] = label
        labels[label] += 1
    # if (sum(labels.values())>=n_shot*7):
    #     break
    else:
        test_outputs[path] = label

print(f"Training set size is ={len(train_outputs)}")
print(f"Testing set size is ={len(test_outputs)}")
print(labels)
fpout = open(f'train_real_{n_shot}_shot.csv', 'w')
fpout.write('path,label\n')

for name, cid in train_outputs.items():
    line = '%s,%s\n' % (name,cid)
    fpout.write(line)
fpout.close() 

fpout = open(f'test_real_{n_shot}_shot.csv', 'w')
fpout.write('path,label\n')

for name, cid in test_outputs.items():
    line = '%s,%s\n' % (name,cid)
    fpout.write(line)
fpout.close() 
        
