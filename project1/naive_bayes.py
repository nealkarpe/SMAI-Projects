from PIL import Image
import numpy as np
import sys

def flatten_image(image):
	out = np.array([])
	for row in image:
		out = np.append(out, row)
	return out

def unflatten_vector(image_vector, height, width):
	out = []
	for i in range(height):
		out.append(image_vector[width*i:width*(i+1)])
	out = np.array(out)
	return out

def PCA(image_vectors, K):
	if K > len(image_vectors):
		K = len(image_vectors)
	A = np.array(image_vectors).T
	eig_vals, eig_vecs = np.linalg.eigh(np.matmul(A.T,A))
	class Eig():
		def __init__(self, val, vec):
			self.val = val
			self.vec = vec
	eig_list = []
	for i in range(len(eig_vals)):
		eig_list.append(Eig(eig_vals[i], eig_vecs[:,i]))
	eig_list.sort(key=lambda x:x.val, reverse=True)
	for eig in eig_list:
		eig_vec = eig.vec
		eig_vec = np.dot(A,eig_vec)
		eig.vec = eig_vec/np.linalg.norm(eig_vec)
	return np.array([np.array(eig_list[i].vec) for i in range(K)])

def comp_reduction(PC, image_vector):
	'''PC is K*d, image_vector is d*1, returns k*1 reduced image vector'''
	return np.dot(PC, image_vector)

def comp_expansion(PC, coefficients):
	'''convert from PC domain to regular domain'''
	total = np.array([0.0]*PC.shape[1])
	for i in range(len(PC)):
		total += coefficients[i]*PC[i]
	return total

def input(image_filenames, class_strings):
	image_vectors = []
	class_label = []
	for i in range(len(image_filenames)):
		with Image.open(image_filenames[i]) as img: 
		    image = np.asarray(img.convert('L'))
		    height, width = image.shape
		    image_vector = flatten_image(image)
		    image_vectors.append(image_vector)
		    class_label.append(class_strings[i])
	avg_vector = sum(image_vectors)/len(image_vectors)
	centered_image_vectors = [(image_vector-avg_vector) for image_vector in image_vectors]
	return image_vectors, centered_image_vectors, height, width, class_label

def findMeans(classes):
	means = {}
	for label in classes:
		classpoints = np.array(classes[label])
		means[label] = sum(classpoints)/len(classpoints)
	return means

def findStds(classes):
	stds = {}
	for label in classes:
		class_std = []
		classpoints = np.array(classes[label])
		num_dimensions = classpoints.shape[1]
		for d in range(num_dimensions):
			class_std.append(np.std(classpoints[:,d]))
		stds[label] = np.array(class_std)
	return stds

def likelihood(test_sample, mean, std):
	prod = 1
	num_dimensions = len(test_sample)
	for d in range(num_dimensions):
		x = test_sample[d]
		try:
			prod *= (1.0/np.sqrt(2*np.pi*std[d]**2))*np.exp(-1.0*(x-mean[d])**2/(2*std[d]**2))
		except:
			return 0
	return prod

def predict(test_sample, classes, prior, means, stds):
	max_posterior = -1
	out = ""
	for label in classes:
		posterior = prior[label]*likelihood(test_sample, means[label], stds[label])
		if posterior > max_posterior:
			max_posterior = posterior
			out = label
	return out

image_filenames = []
class_strings = []
with open(sys.argv[1]) as f:
	for line in f:
		image_filename, class_label = line.split()
		image_filenames.append(image_filename)
		class_strings.append(class_label)

image_vectors, centered_image_vectors, height, width, class_label = input(image_filenames, class_strings)
PC = PCA(centered_image_vectors,32) # K*d matrix of principal components
reduced_image_vectors = []
classes = {}
for i in range(len(image_vectors)):
	row = image_vectors[i]
	coefficients = comp_reduction(PC, row)
	reduced_image_vectors.append(coefficients)
	label = class_label[i]
	if label not in classes:
		classes[label] = [coefficients]
	else:
		classes[label].append(coefficients)

prior = {}
for label in classes:
	prior[label] = len(classes[label])/len(reduced_image_vectors)
means = findMeans(classes)
stds = findStds(classes)

with open(sys.argv[2]) as f:
	for line in f:
		test_image = line.split()[0]
		with Image.open(test_image) as img:
		    image = np.asarray(img.convert('L'))
		    image_vector = flatten_image(image)
		    reduced_image_vector = comp_reduction(PC, image_vector)
		    pred = predict(reduced_image_vector, classes, prior, means, stds)
		    print(pred)
