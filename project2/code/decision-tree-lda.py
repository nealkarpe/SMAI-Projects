import subprocess
import matplotlib.pyplot as plt

accuracies = []
max_depths = list(range(1,31))

for i in range(1,31):
	command_string = "python3 run.py --data_representation lda --classifier_type decision-tree --max_depth " + str(i)
	command_arr = command_string.split()
	proc = subprocess.Popen(command_arr, stdout=subprocess.PIPE)
	(out, err) = proc.communicate()
	out = out.decode("utf-8")
	accuracy = float(out.split("\n")[-2])
	accuracies.append(accuracy)
	print(i, accuracy)

print("max accuracy", max(accuracies))

plt.plot(max_depths,accuracies,color='gold',marker='o',zorder=1)
plt.scatter([7],[0.2348],color="darkseagreen",marker="*",s=150,label="best accuracy",zorder=2)
plt.legend()
plt.title("[LDA] Accuracy of test data vs. max_depth of decision tree")
plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.show()
