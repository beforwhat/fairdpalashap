import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('FedADDP隐私预算与损失关系')
plt.xlabel('隐私预算ε')
plt.ylabel('模型损失')
plt.plot([1,4,8,16], [15,8.5,4.8,2.5])
plt.savefig('test_chinese.png')