import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import grad
import datetime
from scipy.optimize import linprog
from tqdm import tqdm
import numpy as np 
from sklearn.linear_model import LogisticRegression, LinearRegression
from utils.other_utils import recourse_validity

#Functions to compute the cost of recourses
def l1_cost(xs, rs):
	cost = []
	for x,r in zip(xs,rs):
		cost.append(np.linalg.norm(r-x,1))
	return np.mean(np.array(cost))

class GermanSCM():
	def __init__(self, X):
		self.f3 = LinearRegression()
		self.f4 = LinearRegression()
		self.personal_status_sex_cols = [c for c in list(X) if "personal_status_sex" in c]
		self.f3.fit(X[self.personal_status_sex_cols+['age']], X['amount'])
		self.f4.fit(X[['amount']], X['duration'])
		self.idx_map = {c:i for i,c in enumerate(list(X))}
		

	def act(self, x_og, grad):
		rec = np.zeros(len(x_og))

		x_og = x_og.flatten()

		u1 = x_og[[self.idx_map[c] for c in self.personal_status_sex_cols]]
		u2 = x_og[self.idx_map["age"]]
		u3 = x_og[self.idx_map["amount"]] - self.f3.predict(
			[x_og[[self.idx_map[c] for c in self.personal_status_sex_cols]+[self.idx_map["age"]]]])[0]
		u4 = x_og[self.idx_map["duration"]] - self.f4.predict([[x_og[self.idx_map["age"]]]])[0]
		
		grad = grad.flatten()

		#u1 is immutable so a1 is not actionable
		a2 = x_og[self.idx_map["age"]]+grad[self.idx_map["age"]]
		a3 = x_og[self.idx_map["amount"]]+grad[self.idx_map["amount"]]
		a4 = x_og[self.idx_map["duration"]]+grad[self.idx_map["duration"]]
		
		x1 = u1 
		if grad[self.idx_map["age"]]>0:
			x2 = a2
		else:
			x2 = u2
		x3 = a3
		x4 = a4

		rec[[self.idx_map[c] for c in self.personal_status_sex_cols]] = x1
		rec[self.idx_map["age"]] = x2
		rec[self.idx_map["amount"]] = x3
		rec[self.idx_map["duration"]] = x4

		return rec

class CausalRecourse():
	def __init__(self, X_train, predict_proba_fn, torch_model,
				step_size=-1e2, 
				lamb=1, 
				feature_costs=None,
				max_iter=100,
				threshold=0.5,
				target = 1.0):
		self.scm = GermanSCM(X_train)
		self.predict_proba_fn = predict_proba_fn
		self.torch_model = torch_model
		self.step_size=step_size
		self.lamb = lamb
		self.max_iter = 100
		self.y_target = target
		self.feature_costs = feature_costs
		if self.feature_costs is not None:
			self.feature_costs = torch.from_numpy(feature_costs).float()


	def get_recourse(self, x):
		rec = x 
		it=0
		while it<self.max_iter:
			grad = self.get_grad(x, rec)
			nrec = self.scm.act(x, grad*self.step_size)
			rec = nrec
			# print(rec)
			f_rec = self.predict_proba_fn(rec)[0]
			# print(f_rec)
			if f_rec>=0.5:
				return rec
			it+=1
		return rec

	def get_grad(self, x, rec):
		x = torch.from_numpy(x).float()
		rec = Variable(torch.from_numpy(rec).float(), requires_grad=True)
		f_x_new = self.torch_model(rec)
		if self.feature_costs is None:
			cost = torch.dist(rec, x, 1)
		else:
			cost = torch.norm(self.feature_costs*(rec-x), 1)
		obj = self.lamb*(f_x_new-self.y_target)**2 + cost
		obj.backward()
		return rec.grad.detach().numpy()

	def choose_params(self, recourse_needed_X, predict_fn):
		step_sizes = [-1e-1, -1, -10]
		lambdas = [1e-1, 1, 10]

		m1_validity = np.zeros((3,3))
		costs = np.zeros((3,3))

		for si,s in enumerate(step_sizes):
			for li, l in enumerate(lambdas):
				print("Testing step size %f, lambda %f" % (s,l))
				recourses = []
				for xi, x in tqdm(enumerate(recourse_needed_X), total=len(recourse_needed_X)):
					self.step_size = s
					self.lamb = l
					r = self.get_recourse(x)
					recourses.append(r)
				v = recourse_validity(predict_fn, np.array(recourses), target=self.y_target, delta_max=0.75, originals = recourse_needed_X)
				m1_validity[si][li] = v
				if self.feature_costs is None:
					cost = l1_cost(recourse_needed_X, recourses)
				else:
					cost = pfc_cost(recourse_needed_X, recourses, self.feature_costs)
				costs[si][li] = cost
		
		max_validity = np.amax(m1_validity) 
		cand_indices = m1_validity>=max_validity
		min_cost = np.amin(costs[cand_indices])
		step_size_idxs, lamb_idxs = np.where(costs == min_cost)
		step_size, lamb = step_sizes[step_size_idxs[0]], lambdas[lamb_idxs[0]]

		return step_size, lamb
