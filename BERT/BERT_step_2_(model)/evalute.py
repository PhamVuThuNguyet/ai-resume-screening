import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Visualize the performance of training model on STS benchmark train, dev, test dataset ####
train_dev_result = pd.read_csv('./data/model/eval/similarity_evaluation_HR-Akatsuki-'
                               'dev_results.csv').iloc[-2:, :][['cosine_pearson', 'euclidean_pearson',
                                                                'manhattan_pearson', 'dot_pearson']]
test_result = pd.read_csv('./data/model/similarity_evaluation_'
                          'HR-Akatsuki-test_results.csv')[['cosine_pearson', 'euclidean_pearson',
                                                           'manhattan_pearson', 'dot_pearson']]

result = pd.concat([train_dev_result, test_result])
result['name'] = ["train", "dev", "test"]
plot_result = pd.melt(result, id_vars=['name'])
print(train_dev_result)
print(test_result)
print(result)

plt.figure(figsize=(16, 9))
p = sns.barplot(x="value", y="variable", hue="name", data=plot_result, palette="Blues_r")
p.set_xlabel("Value\n (càng lớn càng tốt)", fontsize=16)
p.set_ylabel("Variable", fontsize=16)
p.set_title("Bảng hệ số tương quan Pearson ứng với các cặp CV-JD", fontsize=16)
plt.legend(loc='best')
plt.show()
