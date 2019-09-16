import matplotlib.pyplot as plt

x = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
# y = [0.3767, 0.3801, 0.3762, 0.3773, 0.3761]
y = []
for solver in x:
	command_string = "python3 run.py --classifier_type logistic-regression --solver " + solver
	command_arr = command_string.split()
	proc = subprocess.Popen(command_arr, stdout=subprocess.PIPE)
	(out, err) = proc.communicate()
	out = out.decode("utf-8")
	accuracy = float(out.split("\n")[-2])
	y.append(accuracy)
	print(solver, accuracy)

x.reverse()
y.reverse()

fig, ax = plt.subplots()
width = 0.5 # the width of the bars 
ax.barh(x, y, width, color="lightgreen")
plt.title('Testing accuracy of Logistic Regression classifier with different solvers')
plt.ylabel('Solver',size=12)
plt.xlabel('Accuracy',size=12)
for i, v in enumerate(y):
    ax.text(v + 0.001, i, str(v), color='blue', fontweight='bold', size=14)
plt.show()