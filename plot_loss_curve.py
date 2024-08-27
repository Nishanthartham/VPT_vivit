import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np 

filename = "VPT-DEEP-SNR-Pretrained-fullDS"

data = [
{'loss': 0.4955, 'grad_norm': 12.82339096069336, 'learning_rate': 1.8e-05, 'epoch': 2.0}
{'loss': 0.3966, 'grad_norm': 11.826650619506836, 'learning_rate': 1.7004761904761905e-05, 'epoch': 3.0}
{'loss': 0.3241, 'grad_norm': 17.9631290435791, 'learning_rate': 1.6004761904761905e-05, 'epoch': 4.0}
{'loss': 0.2822, 'grad_norm': 9.487351417541504, 'learning_rate': 1.5004761904761906e-05, 'epoch': 5.0}
{'loss': 0.248, 'grad_norm': 10.579813957214355, 'learning_rate': 1.4004761904761905e-05, 'epoch': 6.0}
{'loss': 0.2047, 'grad_norm': 5.380438804626465, 'learning_rate': 1.3004761904761906e-05, 'epoch': 7.0}
{'loss': 0.1949, 'grad_norm': 22.580047607421875, 'learning_rate': 1.2004761904761906e-05, 'epoch': 8.0}
{'loss': 0.1867, 'grad_norm': 1.1735789775848389, 'learning_rate': 1.1004761904761905e-05, 'epoch': 9.0}
{'loss': 0.1434, 'grad_norm': 0.47216933965682983, 'learning_rate': 1.0004761904761906e-05, 'epoch': 10.0}
{'loss': 0.1301, 'grad_norm': 9.796244621276855, 'learning_rate': 9.004761904761906e-06, 'epoch': 11.0}
{'loss': 0.1229, 'grad_norm': 0.23805901408195496, 'learning_rate': 8.004761904761905e-06, 'epoch': 12.0}
{'loss': 0.093, 'grad_norm': 19.53717613220215, 'learning_rate': 7.00952380952381e-06, 'epoch': 13.0}
{'loss': 0.0951, 'grad_norm': 0.020402567461133003, 'learning_rate': 6.0095238095238095e-06, 'epoch': 14.0}
{'loss': 0.0743, 'grad_norm': 0.6684430837631226, 'learning_rate': 5.00952380952381e-06, 'epoch': 15.0}
{'loss': 0.0663, 'grad_norm': 14.92542552947998, 'learning_rate': 4.00952380952381e-06, 'epoch': 16.0}
{'loss': 0.0677, 'grad_norm': 26.08776092529297, 'learning_rate': 3.00952380952381e-06, 'epoch': 17.0}
{'loss': 0.0527, 'grad_norm': 17.729093551635742, 'learning_rate': 2.0095238095238097e-06, 'epoch': 18.0}
{'loss': 0.0344, 'grad_norm': 0.04024840146303177, 'learning_rate': 1.0095238095238095e-06, 'epoch': 19.0}
{'loss': 0.0378, 'grad_norm': 0.011689453385770321, 'learning_rate': 9.523809523809524e-09, 'epoch': 20.0}
]

epoches = [d['epoch'] for d in data]
losses = [d['loss'] for d in data]

# plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
plt.plot(epoches,losses,color='blue',label='Train Loss')
# plt.setp(line)
plt.xticks(range(1,len(data)+1,1))
plt.yticks(np.arange(0,max(losses)+0.2,0.2))
plt.xlabel("Epoches")
plt.ylabel("Train loss")
plt.title("VPT Deep finetuning on SNR pretrained model on complete dataset")
plt.legend()
plt.savefig(f"plots/prompting_complete_training/{filename}.png")
