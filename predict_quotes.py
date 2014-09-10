from __future__ import division, print_function
import numpy as np 
import math
from sklearn.utils import check_random_state
from sklearn import metrics as skm

# various prediction approaches.

'''
	For each prediction method here:

	__init__ reads in parameters.

	fit reads in complete columns of the domain-quote binary matrix.

	predict takes binary domain-quote columns. for each column, it 
	hides one entry at a time and makes a prediction on that entry given the 
	other values in the column accordingly.
'''

class BerPred(object):

	'''
		Suppose that n out of N possible (domain,quote) pairs occur. Then
		the probability that a new domain cites a new quote is n/N. So citation
		is a Bernoulli RV. 

		A baseline.
	'''

	def __init__(self, random_state = 0):

		self.rng = check_random_state(random_state)

	def fit(self, X):

		total_cites = np.sum(X)
		total_possible = np.dot(*X.shape)

		self.num_domains = X.shape[0]

		self.cite_prob = total_cites / total_possible

	def predict(self, X=None):

		return self.rng.rand(self.num_domains, 1) <= self.cite_prob

class DomainBerPred(object):

	'''
		Suppose domain D cites n out of N total quotes. 
		Then D will cite a new quote q with probability n/N. 

		A baseline.

	'''

	def __init__(self, random_state = 0):

		self.rng = check_random_state(random_state)

	def fit(self, X):

		num_quotes = X.shape[1]
		
		self.cite_probs = np.sum(X, axis=1) / num_quotes
		self.cite_probs = self.cite_probs[:,np.newaxis]

	def predict(self, X = None):

		shape = self.cite_probs.shape
		if X is not None:
			shape = X.shape

		return self.rng.rand(*shape) <= self.cite_probs


class QuoteBerPred(object):

	'''
		Suppose that n out of N domains cite a quote q. Then a new
		domain d will cite q with probability n/N. 

		A baseline.
	'''

	def __init__(self, random_state = 0):

		self.rng = check_random_state(random_state)

	def fit(self, X):

		pass

	def predict(self, X):

		num_domains = X.shape[0]

		cite_counts = np.array([np.sum(X, axis = 0)] * num_domains)
		cite_counts -= X

		cite_probs = cite_counts/(num_domains - 1)

		return self.rng.rand(*X.shape) <= cite_probs

class QuoteSamplePred(object):

	'''
		Samples 50% of the domains and sees what proportion p cites a quote q.
		Then a new domain will cite q if p is above some threshold.
	'''

	def __init__(self, threshold=0.5, random_state = 0, num_samples = 5,
					proportion_samples = .5):

		self.threshold = threshold
		self.rng = check_random_state(random_state)
		self.num_samples = num_samples
		self.proportion_samples = proportion_samples

	def fit(self, X=None):

		pass

	def predict(self, X):

		num_domains = X.shape[0]
		num_to_sample = math.floor(num_domains * self.proportion_samples)

		prob_array = np.ones(num_domains,)
		prob_array *= 1 / (num_domains - 1)

		predictions = np.zeros(X.shape, dtype=bool)

		for q in range(X.shape[1]):
			for d in range(num_domains):
				prob_array[d] = 0

				num_nonzero = 0

				for i in range(self.num_samples):
					samples = self.rng.choice(X[:,q], num_to_sample, p=prob_array)
					num_nonzero += np.count_nonzero(samples)

				predictions[d,q] = (num_nonzero / (self.num_samples * num_to_sample)) >= self.threshold

				prob_array[d] = 1 / (num_domains -1)

		return predictions

class RandomPred(object):

	'''
		generates x ~Uni(0,1) and sees if this is above a threshold.
	'''

	def __init__(self, threshold = 0.5, random_state = 0):

		self.threshold = threshold
		self.rng = check_random_state(random_state)

	def fit(self, X=None):

		pass

	def predict(self, X):

		return self.rng.rand(*X.shape)  >= self.threshold

class WeightedVotePred(object):

	'''
		Let X = (x_ij) be the binary matrix co-occurrence matrix
			( where x_ij = 1(domain i cites quote j) )
		and X_norm be X normalized by column and row;  
		similarity S = (s_ii') = X X.T.

		For domain i and quote j let 
			p_ij = \sum(i' != i) s_ii' x_i'j / \sum(i' != i) s_ii' (not sum x_i'j...)

		Then i quotes j if p_ij >= t, where t is some threshold.
	'''

	def __init__(self, threshold = 0.5):

		self.threshold = threshold

	def fit(self, X):

		col_sums = np.sum(X, axis=0)
		col_sums[col_sums == 0] = 1

		norm_by_col = X / np.sqrt(col_sums)

		row_norms = np.linalg.norm(norm_by_col, axis=1)
		row_norms[row_norms == 0] = 1

		norm_by_row = norm_by_col / row_norms[:, np.newaxis]

		self.sim = np.dot(norm_by_row, norm_by_row.T)
		np.fill_diagonal(self.sim, 0)

		self.sim_normed_by_row = self.sim / np.sum(self.sim, axis=1)[:, np.newaxis]

	def predict(self, X):

		pred_scores = np.dot(self.sim_normed_by_row, X)
		'''num_domains = X.shape[0]

		quote_counts = np.array([np.sum(X, axis=0)] * num_domains)
		quote_counts -= X 
		quote_counts[quote_counts == 0] = 1

		pred_scores = np.dot(self.sim, X)
		pred_scores /= quote_counts'''

		return pred_scores >= self.threshold

class ClassWeightedPred(object):

	'''
		Let similarity S be defined as in WeightedVotePred. 

		We give a weight w to quotes omitted by a domain. Then for 
		domain i and quote j, let

			p_ij = \sum(i' != i, X_i'j = 1) s_ii' - w * \sum(i' != i, X_i'j = 0) s_ii'

		If to_normalize is set to True, then we divide each term by the number of
		domains which cited, or didn't cite, the quote respectively.

	'''

	def __init__(self, weight = -1, to_normalize = False):

		self.weight = weight

		if self.weight > 0:
			self.weight *= -1 # because I will probably pass in a positive weight...

		self.to_normalize = to_normalize

	def fit(self, X):

		col_sums = np.sum(X, axis=0)
		col_sums[col_sums == 0] = 1

		norm_by_col = X / np.sqrt(col_sums)

		row_norms = np.linalg.norm(norm_by_col, axis=1)
		row_norms[row_norms == 0] = 1

		norm_by_row = norm_by_col / row_norms[:, np.newaxis]

		self.sim = np.dot(norm_by_row, norm_by_row.T)
		np.fill_diagonal(self.sim, 0)

	def predict(self, X):

		num_domains = X.shape[0]

		if self.to_normalize:

			X_pos = X.copy()
			X_neg = 1 - X

			pos_counts = np.array([np.sum(X_pos, axis=0)] * num_domains)
			pos_counts -= X_pos
			pos_counts[pos_counts == 0] = 1

			pos_component_scores = np.dot(self.sim, X_pos)
			pos_component_scores /= pos_counts

			neg_counts = np.array([np.sum(X_neg, axis=0)] * num_domains)
			neg_counts -= X_neg
			neg_counts[neg_counts == 0] = 1
			
			neg_component_scores = np.dot(self.sim, X_neg)
			neg_component_scores /= neg_counts
			neg_component_scores *= self.weight

			combined_scores = pos_component_scores + neg_component_scores

			return combined_scores >= 0

		else:

			X_weighted = X.copy()
			X_weighted[X_weighted == 0] = self.weight

			pred_scores = np.dot(self.sim, X_weighted)

			return pred_scores >= 0

class PredictResult(object):

	def __init__(self, true, pred):

		self.true = true
		self.pred = pred

		self.scores = {}
		self.eval_scores()

	def eval_scores(self):

		self.scores['accuracy'] = np.count_nonzero(self.true == self.pred) / np.dot(*self.true.shape)
		accuracies = np.zeros(self.true.shape[1])
		supports = np.sum(self.true, axis=0)
		for i in range(self.true.shape[1]):
			accuracies[i] = skm.accuracy_score(self.true[:,i], self.pred[:,i])
		
		self.scores['weighted_accuracy'] = np.average(accuracies, weights=supports)

		self.scores['micro_precision'] = skm.precision_score(self.true, self.pred, average='micro')
		self.scores['macro_precision'] = skm.precision_score(self.true, self.pred, average='macro')
		self.scores['weighted_precision'] = skm.precision_score(self.true, self.pred, average='weighted')
		self.scores['micro_recall'] = skm.recall_score(self.true, self.pred, average='micro')
		self.scores['macro_recall'] = skm.recall_score(self.true, self.pred, average='macro')
		self.scores['weighted_recall'] = skm.recall_score(self.true, self.pred, average='weighted')
		self.scores['micro_f1'] = skm.f1_score(self.true, self.pred, average='micro')
		self.scores['macro_f1'] = skm.f1_score(self.true, self.pred, average='macro')
		self.scores['weighted_f1'] = skm.f1_score(self.true, self.pred, average='weighted')

class WeightedVoteTopicPred(object):
	'''	
		Weighted vote with topics.
		For each topic, we compute a topic similarity between domains by replacing entries=1 in the domain-quote 
			binary matrix with the topic weight of that quote as given by LDA, and then using our similarity
			measure as before.
		For a new domain D and quote Q, score is given by
		\sum_t in topics w(t,Q) \sum_d != D 1(d,Q)sim_t(d,D)/\sum_t w(t,Q) \sum_d != D sim_t(d,D)
	''' 
    def __init__(self, threshold=0.5):
        
        self.threshold = threshold
        
        
    def fit(self, X, topic_weights):
        
        num_domains = X.shape[0]
        
        num_topics = topic_weights.shape[0]
        self.sims_by_topic = []
                
        for i in range(num_topics):
            
            topic_matrix = get_topic_matrix(X, i, topic_weights)
            
            col_sums = np.sum(topic_matrix, axis=0)
            col_sums[col_sums == 0] = 1 
            
            norm_by_col = topic_matrix / np.sqrt(col_sums)
            
            row_norms = np.linalg.norm(norm_by_col, axis=1)
            row_norms[row_norms == 0] = 1
            
            norm_by_row = norm_by_col / row_norms[:, np.newaxis]
            
            topic_sim = np.dot(norm_by_row, norm_by_row.T)
            np.fill_diagonal(topic_sim, 0)
            
            self.sims_by_topic.append(topic_sim)
        
    
    def predict(self, X, topic_weights):
        
        num_domains = X.shape[0]
        pred_scores = np.zeros((num_domains,1), dtype=float)
        
        norm_constants = np.zeros((num_domains,))
        for i in range(len(self.sims_by_topic)):
            
            pred_scores += (topic_weights[i] * np.dot(self.sims_by_topic[i],X))        
            norm_constants += (topic_weights[i] * np.sum(self.sims_by_topic[i], axis=1))
            
        pred_scores = np.squeeze(pred_scores) / norm_constants
                
        return pred_scores >= self.threshold
            

def eval_vote_model(model_name, data, random_state = 0, **kwargs):
	'''
		evaluates model with leave one out.
		arguments:
			model_name: model to use.
			data: binary matrix of domains to quotes
			
		returns: 
			matrix of predictions, with one entry per domain,quote pair
	''' 
    if model_name == 'BerPred':
        model = BerPred(random_state = random_state)
    elif model_name == 'DomainBerPred':
        model = DomainBerPred(random_state = random_state)
    elif model_name == 'QuoteBerPred':
        model = QuoteBerPred(random_state = random_state)
    elif model_name == 'QuoteSamplePred':
    	model = QuoteSamplePred(threshold = kwargs['threshold'], random_state = random_state)
    elif model_name == 'RandomPred':
    	model = RandomPred(threshold = kwargs['threshold'], random_state = random_state)
    elif model_name == 'WeightedVotePred':
        model = WeightedVotePred(threshold = kwargs['threshold'])
    elif model_name == 'ClassWeightedPred':
        model = ClassWeightedPred(weight = kwargs['weight'], to_normalize=kwargs['to_normalize'])
    
    num_cols = data.shape[1]
    mask = np.ones(num_cols, dtype=np.bool)
    
    preds = np.zeros(data.shape)
    
    for i in range(num_cols):
        
        mask[i] = False
        
        model.fit(data[:,mask])
        
        pred = model.predict(data[:,~mask])
        preds[:,[i]] = pred

        mask[i] = True

        
    return preds

def eval_mc(matrix, learnrate, alpha,tolerance=5e-2):
	'''
		evaluates matrix completion predictor with leave one out
	'''
    pred = np.zeros(matrix.shape)
    matrix_mc = matrix.copy()
    matrix_mc[matrix_mc == 0] = -1
    for i in range(matrix_mc.shape[0]):
        for j in range(matrix_mc.shape[1]):
            mc = MatrixCompletion(method='sgd',is_classification=True,classification_loss='logloss',
                shuffle=True,initial_learning_rate=learnrate,verbose=None, tol=tolerance,alpha=alpha)
            rmask = [i]
            cmask = [j]
            mask = np.zeros(matrix.shape, dtype=np.bool)
            mask[rmask,cmask] = 1
            fit_mask = (~mask).nonzero()
            mc.fit(matrix_mc,mask=fit_mask)
            pred[i,j] = np.sign(np.dot(mc.U_,mc.V_.T))[i,j]
    pred[pred == -1] = 0
    return pred

def eval_wv_topic_model(data, topic_weights, random_state = 0, threshold = 0.5):
	'''
		evaluates weighted vote with topic predictor, using leave one out.
	'''
    model = WeightedVoteTopicPred(threshold = threshold)
    
    num_cols = data.shape[1]
    mask = np.ones(num_cols, dtype=np.bool)
    
    preds = np.zeros(data.shape)
    
    
    for i in range(num_cols):
        
        mask[i] = False
        model.fit(data[:,mask], topic_weights[:,mask])
        
        pred = model.predict(data[:,~mask], topic_weights[:,~mask])
        preds[:,i] = pred

        mask[i] = True
    
    return preds
