import matplotlib.pyplot as plt

C_vals = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]

# acc_vals = [0.27944, 0.27849, 0.27897, 0.27917, 0.27852, 0.27984, 0.27927, 0.27874, 0.27851, 0.27872, 0.27799, 0.28010, 0.27843, 0.27978, 0.27954, 0.27903]
acc_vals = []
for c in C_vals:
	command_string = "python3 run.py --classifier_type linear-svm --C " + str(c)
	command_arr = command_string.split()
	proc = subprocess.Popen(command_arr, stdout=subprocess.PIPE)
	(out, err) = proc.communicate()
	out = out.decode("utf-8")
	accuracy = float(out.split("\n")[-2])
	acc_vals.append(accuracy)
	print(c, accuracy)

plt.plot(C_vals,acc_vals,color='orangered',marker='o')
plt.title("Testing accuracy vs. 'C' hyperparameter value")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.show()