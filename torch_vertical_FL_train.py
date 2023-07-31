# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 09:43:55 2021

@author: adamwei
"""
import argparse
import time
import pandas as pd
from thop import profile
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score
# https://imbalanced-learn.org/stable/
from avazudataset import AvazuDataset
from utils import load_dat, batch_split, create_avazu_dataset
from captum.attr import IntegratedGradients, ShapleyValueSampling, GradientShap, Occlusion, FeatureAblation, InputXGradient,\
	KernelShap
# from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients, \
#     NeuronConductance, NeuronIntegratedGradients, Saliency, DeepLift, DeepLiftShap, \
#     FeatureAblation, FeaturePermutation, GradientShap, GuidedBackprop, GuidedGradCam, \
#     InputXGradient, Occlusion, ShapleyValueSampling, SigmoidGrad, SmoothGrad, \
#     Deconvolution, GradientShap, GuidedBackprop, GuidedGradCam, InputXGradient, \
#     LayerConductance, LayerIntegratedGradients, NeuronConductance, NeuronIntegratedGradients, \
#     Saliency, DeepLift, DeepLiftShap, FeatureAblation, FeaturePermutation, GradientShap
		

"""
Reference
	https://www.kaggle.com/c/avazu-ctr-prediction
"""
from torch_model import MlpModel, torch_organization_model, torch_top_model


def plot_contributions(data_dict, epoch):
	import matplotlib.pyplot as plt

	# Create a figure and axis
	fig, ax = plt.subplots()

	# Define a list of colors for the lines
	colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan',
			  'magenta', 'yellow', 'lime', 'skyblue', 'olive', 'teal', 'salmon', 'lavender',
			  'peru', 'indigo', 'gold', 'steelblue', 'tomato', 'limegreen', 'slategray']

	# Plot each line chart with a specific color and label
	for idx, (key, value) in enumerate(data_dict.items()):
		color = colors[idx % len(colors)]  # Use modulo to cycle through colors if more keys than colors
		ax.plot(value, label=str(key), color=color)

	# Set labels and title
	ax.set_xlabel('')
	ax.set_ylabel('Percent Contribution')
	ax.set_title(f'Percent Contribution for Clients in epoch {epoch}')
	ax.legend()

	plt.savefig(f'percent_contributions_performance_{epoch}')
	# Show the plot
	plt.show()



def main(args):
	
	data_type = args.data_type                  # define the data options: 'original', 'encoded'
	model_type = args.model_type                # define the learning methods: 'vertical', 'centralized'
	epochs = args.epochs                        # number of training epochs
	organization_num = args.organization_num    # number of participants in vertical FL
	attribute_split_array = \
		np.zeros(organization_num).astype(int)  # initialize a dummy split scheme of attributes
	nrows = 50000                                # subselection of rows to speed up the program
	
	# dataset preprocessing
	if data_type == 'original': 
		
		if args.dname == 'ADULT':
			
			file_path = "./datasets/{0}.csv".format(args.dname)
			X = pd.read_csv(file_path)
			# X = pd.read_csv(file_path, nrows = nrows)
			
			# Ming added the following codes on 11/11/2021 to pre-process the Adult dataset
			# remove a duplicated attributed of "edu"
			X = X.drop('edu', axis=1)
			# rename an attribute from "skin" to "race"
			X = X.rename({'skin':'race'}, axis=1)
 
			# Ming added the following codes on 11/11/2021 to perform a quick sanity check of the dataset
			X.head()
			for attribute in X.columns:
				#print(dt.value_counts(dt[attribute]))
				plt.figure()
				X.value_counts(X[attribute]).sort_index(ascending=True).plot(kind='bar')
			 
			# get the attribute data and label data
			# y = X['income'].values.astype('int')
			y = X['income'].apply(lambda x: bool(">" in x)).astype("int")
			X = X.drop(['income'], axis=1)
			print(X)
			
			if args.attack != "originial":
			# Generate an array of random categories
			  noisy_categories = np.random.choice(['A', 'B', 'C'], size=[X.shape[0], 3])
			  
			  # Convert the 2D array to a 1D array
			  noisy_categories_flat = noisy_categories.flatten()
			  
			  # Convert the categories into a categorical data type
			  noisy_categories_lst = pd.Categorical(noisy_categories_flat)

			  # Reshape the 1D array back to a 2D array
			  noisy_categories_2d = noisy_categories_lst.codes.reshape(-1, 3)

			  # print('noisy_categories_lst: ', noisy_categories_2d)
			  # Add the categories to the DataFrame
			  X = pd.concat([X.reset_index(drop=True), pd.DataFrame(noisy_categories_2d)], axis=1)
			N, dim = X.shape
					   
			#add 6 more random value columns to the dataset
			# X = pd.concat([X, pd.DataFrame(np.random.randn(N, 7))], axis=1)
			   
			 # Print out the dataset settings
			print("\n\n=================================")
			print("\nDataset:", args.dname, \
				   "\nNumber of attributes:", dim-1, \
				   "\nNumber of labels:", 1, \
				   "\nNumber of rows:", N,\
				   "\nPostive ratio:", sum(y)/len(y))     
			   
			columns = list(X.columns)
			 
			 # set up the attribute split scheme for vertical FL
			attribute_split_array = \
				 np.ones(len(attribute_split_array)).astype(int) * \
				 int(dim/organization_num)
			print(attribute_split_array.shape)
			 # correct the attribute split scheme if the total attribute number is larger than the actual attribute number
			if np.sum(attribute_split_array) > dim:
				print('unknown error in attribute splitting!')
			elif np.sum(attribute_split_array) < dim:
				missing_attribute_num = (dim) - np.sum(attribute_split_array)
				attribute_split_array[-1] = attribute_split_array[-1] + missing_attribute_num
			else:
				print('Successful attribute split for multiple organizations')
	 
		elif args.dname == 'AVAZU':
			
			file_path = './datasets/{0}.gz'.format(args.dname)
			df = pd.read_csv(file_path, compression='gzip', nrows=nrows)
			df.to_csv('./datasets/{0}.csv'.format(args.dname), index=False)
			
			columns = df.columns.drop('id')
			# data = pd.read_csv(file_path, compression='gzip', nrows=nrows)
			data = AvazuDataset('./datasets/{0}.csv'.format(args.dname), rebuild_cache=True)
			
			# X = data.fillna('-1')
			
			# X.head()

			X, y = [data[i][0] for i in range(len(data))], [data[i][1] for i in range(len(data))]
			y = np.reshape(y, [len(y), 1])
			data = np.concatenate((y, X),axis=1)
			
			X = pd.DataFrame(data, columns=columns)
			
			# get the attribute data and label data
			y = X['click'].values.astype('int')
			X = X.drop(['click'], axis=1)
			
			N, dim = X.shape
			   
			  # Print out the dataset settings
			print("\n\n=================================")
			print("\nDataset:", args.dname, \
					"\nNumber of attributes:", dim-1, \
					"\nNumber of labels:", 1, \
					"\nNumber of rows:", N,\
					"\nPostive ratio:", sum(y)/len(y))            
			
			columns = list(X.columns)
			
			  # set up the attribute split scheme for vertical FL
			attribute_split_array = \
				  np.ones(len(attribute_split_array)).astype(int) * \
				  int(dim/organization_num)
				 
			  # correct the attribute split scheme if the total attribute number is larger than the actual attribute number
			if np.sum(attribute_split_array) > dim:
				print('unknown error in attribute splitting!')
			elif np.sum(attribute_split_array) < dim:
				missing_attribute_num = dim - np.sum(attribute_split_array)
				attribute_split_array[-1] = attribute_split_array[-1] + missing_attribute_num
			else:
				print('Successful attribute split for multiple organizations')
	 
	else:
		file_path = "./dataset/{0}.dat".format(args.dname)
		X, y = load_dat(file_path, minmax=(0, 1), normalize=False, bias_term=True)  
 
	
	# initialize the arrays to be return to the main function
	loss_array = []
	auc_array = []
	test_epoch_array = []
	
	if model_type == 'vertical':
				
		# print out the vertical FL setting 
		print('\nThe current vertical FL has a non-configurable structure.')
		print('Reconfigurable vertical FL can be achieved by simply changing the attribute group split!')
		
		# Ming revised the following codes on 12/11/2021 to realize re-configurable vertical FL
		print('Ming revised the codes on 12/11/2021 to realize re-configurable vertical FL.')       
		print('\nThere are {} participant organizations:'.format(organization_num))
		
		# define the attributes in each group for each organization
		attribute_groups = []
		attribute_start_idx = 0
		for organization_idx in range(organization_num):
			attribute_end_idx = attribute_start_idx + attribute_split_array[organization_idx]
			attribute_groups.append(columns[attribute_start_idx : attribute_end_idx])
			attribute_start_idx = attribute_end_idx
			print('The attributes held by Organization {0}: {1}'.format(organization_idx, attribute_groups[organization_idx]))                        
		
		# get the vertically split data with one-hot encoding for multiple organizations
		vertical_splitted_data = {}
		encoded_vertical_splitted_data = {}
		chy_one_hot_enc = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
		for organization_idx in range(organization_num):
			
			vertical_splitted_data[organization_idx] = \
				X[attribute_groups[organization_idx]].values#.astype('float32')
			encoded_vertical_splitted_data[organization_idx] = \
				chy_one_hot_enc.fit_transform(vertical_splitted_data[organization_idx])
			
			print(vertical_splitted_data)
			print('vertical split data for organization {0}: {1}'.format(organization_idx, vertical_splitted_data[organization_idx]))
			print('encoded vertical split data for organization {0}: {1}'.format(organization_idx, encoded_vertical_splitted_data[organization_idx]))
			print('The shape of the encoded dataset held by Organization {0}: {1}'.format(organization_idx, np.shape(encoded_vertical_splitted_data[organization_idx]))) 
		print('X shape:',X.shape)
		print('attribute groups:',attribute_groups)
		# set up the random seed for dataset split
		random_seed = 1001
		
		# split the encoded data samples into training and test datasets
		X_train_vertical_FL = {}
		X_test_vertical_FL = {}
		
		for organization_idx in range(organization_num):
			if organization_idx == 0:
				X_train_vertical_FL[organization_idx], X_test_vertical_FL[organization_idx], y_train, y_test = \
					train_test_split(encoded_vertical_splitted_data[organization_idx], y, test_size=0.2, random_state=random_seed)
				print(X_test_vertical_FL,X_train_vertical_FL)
			else:
				X_train_vertical_FL[organization_idx], X_test_vertical_FL[organization_idx], _, _ = \
					train_test_split(encoded_vertical_splitted_data[organization_idx], y, test_size=0.2, random_state=random_seed)

		train_loader_list, test_loader_list = [], []
		for organization_idx in range(organization_num):
		
			X_train_vertical_FL[organization_idx] = torch.from_numpy(X_train_vertical_FL[organization_idx]).float()
			X_test_vertical_FL[organization_idx] = torch.from_numpy(X_test_vertical_FL[organization_idx]).float()
			train_loader_list.append(DataLoader(X_train_vertical_FL[organization_idx], batch_size=args.batch_size))
			test_loader_list.append(DataLoader(X_test_vertical_FL[organization_idx], batch_size=len(X_test_vertical_FL[organization_idx]), shuffle=False))
			# train_loader = DataLoader(X_train_vertical_FL[organization_idx], batch_size=args.batch_size)
			# test_loader = DataLoader(X_test_vertical_FL[organization_idx], batch_size=len(X_test_vertical_FL[organization_idx]), shuffle=False)
					  
			
		# y_train = torch.from_numpy(y_train).long()
		# y_test = torch.from_numpy(y_test).long()
		
		y_train = torch.from_numpy(y_train.to_numpy()).float()
		y_test = torch.from_numpy(y_test.to_numpy()).float()
	
		train_loader_list.append(DataLoader(y_train, batch_size=args.batch_size))
		test_loader_list.append(DataLoader(y_test, batch_size=args.batch_size))
  
		# set the neural network structure parameters
		organization_hidden_units_array = [np.array([128])]*organization_num
		organization_output_dim = np.array([64 for i in range(organization_num)])
	
		top_hidden_units = np.array([64])
		top_output_dim = 1
		
		# build the client models
		organization_models = {}
		for organization_idx in range(organization_num):
			organization_models[organization_idx] = \
				torch_organization_model(X_train_vertical_FL[organization_idx].shape[-1],\
								organization_hidden_units_array[organization_idx],
								organization_output_dim[organization_idx])
			 
		# build the top model over the client models
		top_model = torch_top_model(sum(organization_output_dim), top_hidden_units, top_output_dim)

		# define the neural network optimizer
		optimizer = torch.optim.Adam(top_model.parameters(), lr=0.002)
		
		optimizer_organization_list = []
		for organization_idx in range(organization_num):
			
			optimizer_organization_list.append(torch.optim.Adam(organization_models[organization_idx].parameters(), lr=0.002))

		#optimizer = optimizers.SGD(learning_rate=0.002, momentum=0)
	
		# conduct vertical FL
		print('\nStart vertical FL......\n')   
		
		# criterion = nn.CrossEntropyLoss()
		criterion = nn.BCELoss()
	
		top_model.train()
		org_contributions = {i: [] for i in range(organization_num)}
		attr_lst = []

		contributions_array = []
		for i in range(epochs):
			contributions = {}
			batch_idxs_list = batch_split(len(X_train_vertical_FL[0]), args.batch_size, args.batch_type)
			
			for batch_idxs in batch_idxs_list:
				optimizer.zero_grad()
				
				for organization_idx in range(organization_num):
					
					optimizer_organization_list[organization_idx].zero_grad()
				
				organization_outputs = {}
				for organization_idx in range(organization_num):
						organization_outputs[organization_idx] = \
							organization_models[organization_idx](X_train_vertical_FL[organization_idx][batch_idxs])
	
				
				organization_outputs_cat = organization_outputs[0]
				if len(organization_outputs) >= 2:
					for organization_idx in range(1, organization_num):
						organization_outputs_cat = torch.cat((organization_outputs_cat,\
										organization_outputs[organization_idx]), 1)
					
				outputs = top_model(organization_outputs_cat)
				# logits = torch.sigmoid(outputs)
				logits = torch.reshape(outputs, shape=[len(outputs)])
				# print('logits: ',logits)
				# print('y_train[batch_idxs]: ',y_train[batch_idxs])
				loss = criterion(logits, y_train[batch_idxs]) 

				

				#compute loss for all organization's contributions

				for org in range(0, organization_num):
					if org != 0:
						organization_outputs_cat_2 = organization_outputs[0]
					else:
						rows, columns = organization_outputs[0].size()
						padding_tensor = torch.zeros(rows, columns)
						organization_outputs_cat_2 = padding_tensor

					for j in range(1, organization_num):
						if j != org:
							organization_outputs_cat_2 = torch.cat((organization_outputs_cat_2,\
										organization_outputs[j]), 1)
						else:
							rows, columns = organization_outputs[j].size()
							padding_tensor = torch.zeros(rows, columns)
							organization_outputs_cat_2 = torch.cat((organization_outputs_cat_2,\
										padding_tensor), 1)

					outputs_2 = top_model(organization_outputs_cat_2)
					logits_2 = torch.reshape(outputs_2, shape=[len(outputs_2)])
					loss_without = criterion(logits_2, y_train[batch_idxs]) 

					loss_with = loss.detach().numpy()
					loss_without = loss_without.detach().numpy()

					if loss_with != 0:
						net_contribution = (loss_without - loss_with)*100/loss_with
					else:
						net_contribution = loss_without*100

					if org in contributions:
						contributions[org].append(net_contribution)
					else:
						contributions[org] = [net_contribution]
				
				loss.backward()  # backpropagate the loss
				optimizer.step() # adjust parameters based on the calculated gradients 
				
				for organization_idx in range(organization_num):
					
					optimizer_organization_list[organization_idx].step()
   
			# let the program report the simulation progress
			

			plot_contributions(contributions, i)
			contributions_array.append(contributions)
			if (i+1)%1 == 0:
				organization_outputs_for_test = {}
				feature_mask_tensor = None
				feature_mask_tensor_list = []
				for organization_idx in range(organization_num):
					organization_outputs_for_test[organization_idx] = \
						organization_models[organization_idx](X_test_vertical_FL[organization_idx])
					#add torch tensors of organization_outputs_for_test[organization_idx] shape with constant value to feature_mask_tensor
					feature_mask_tensor_list.append(torch.full(organization_outputs_for_test[organization_idx].shape, organization_idx))
				feature_mask_tensor = torch.stack(feature_mask_tensor_list, dim=0)
				# print('feature_mask_tensor shape: ', feature_mask_tensor.shape)
				organization_outputs_for_test_cat = organization_outputs_for_test[0]
				if len(organization_outputs_for_test) >= 2:
					
					for organization_idx in range(1, organization_num):
						organization_outputs_for_test_cat = torch.cat((organization_outputs_for_test_cat,\
										organization_outputs_for_test[organization_idx]), 1)
						# print('organization outputs shape: ', organization_outputs_for_test[organization_idx].shape)
						
						# print('organization outputs shape cat: ', organization_outputs_for_test_cat.shape)
					
				outputs = top_model(organization_outputs_for_test_cat)
				# print('outputs shape: ', outputs)
				# log_probs = torch.sigmoid(outputs)
				log_probs = torch.reshape(outputs, shape=[len(outputs)])
				
				auc = roc_auc_score(y_test, log_probs.data)
				
				
				print('For the {0}-th epoch, train loss: {1}, test auc: {2}'.format(i+1, loss.detach().numpy(), auc))
				
				test_epoch_array.append(i+1)
				loss_array.append(loss.detach().numpy())
				auc_array.append(auc)
				
		# 		contribution_method = args.contribution_schem
				
		# 		if contribution_method == 'ig':
		# 			# create an instance of the attribution method
		# 			ig = IntegratedGradients(top_model)
		# 			X_example = organization_outputs_for_test_cat
		# 			attr, delta = ig.attribute(X_example, baselines=torch.zeros_like(X_example), return_convergence_delta=True, n_steps=10)
					
		# 		elif contribution_method == 'sv':
		# 			sv = ShapleyValueSampling(top_model)
		# 			X_example = organization_outputs_for_test_cat
		# 			attr = sv.attribute(X_example)
					
		# 		elif contribution_method == 'gsv':
		# 			gradient_shap = GradientShap(top_model)
		# 			X_example = organization_outputs_for_test_cat
		# 			attr = gradient_shap.attribute(X_example)
					
		# 		elif contribution_method == 'inputGrad':
		# 			input_x_gradient = InputXGradient(top_model)
		# 			X_example = organization_outputs_for_test_cat
		# 			attr = input_x_gradient.attribute(X_example)
					
		# 		elif contribution_method == 'fa':
		# 			ablator = FeatureAblation(top_model)
		# 			X_example = organization_outputs_for_test_cat
		# 			attr = ablator.attribute(X_example)
					
		# 		elif contribution_method == 'ks':
		# 			ks = KernelShap(top_model)
		# 			X_example = organization_outputs_for_test_cat
		# 			attr = ks.attribute(X_example)
					
		# 		# print the attribution scores for each organization
				
		# 		# print('attr shape: ', attr.shape)
		# 		# print('attr type: ', type(attr))
		# 		# # Compute mean attribution along the first axis (axis=0)
		# 		# feature_attributions = attr.mean(axis=0)

		# 		# # Print feature-wise attribution values
		# 		# for i, attribution in enumerate(feature_attributions):
		# 		#     print(f"Feature {i}: {attribution}")
				
		# #         temp_attr_lst = []
		# #         for organization_idx in range(organization_num):
		# #             organization_attr = attr[:, organization_idx*organization_output_dim[organization_idx]:(organization_idx+1)*organization_output_dim[organization_idx]]
		# #             print('Organization ', organization_idx,' contribution: ', '{:.8f}'.format(organization_attr.mean()))
		# #             temp_attr_lst.append(organization_attr.mean())
		# #         attr_lst.append(temp_attr_lst)
		# #         print('type of attr_lst: ', type(attr_lst[0]))
		# #         #take mean along the first axis
		# #         # print('attributions across epochs: ', np.array([attr.detach().numpy() for attr in attr_lst]).mean(axis=1))
		# # print('attributions across epochs: ', attr_lst)
		# 		temp_attr_lst = []
		# 		for organization_idx in range(organization_num):
		# 			organization_attr = attr[:, organization_idx*organization_output_dim[organization_idx]:(organization_idx+1)*organization_output_dim[organization_idx]]
		# 			org_contributions[organization_idx].append(organization_attr.mean())
		# 			print('Organization ', organization_idx,' contribution: ', '{:.8f}'.format(organization_attr.mean()))
		# 			temp_attr_lst.append(organization_attr.mean())
		# 		attr_lst.append(temp_attr_lst)
				
		# 		print('type of attr_lst: ', type(attr_lst[0]))

		# fig, ax = plt.subplots()
		# contributions_per_epoch = []

		# for organization_idx in range(organization_num):
		# 	org_contributions_tensor = torch.tensor(org_contributions[organization_idx])
		# 	org_contributions_np = org_contributions_tensor.detach().numpy()
		# 	contributions_per_epoch.append(org_contributions_np)

		# total_contributions_per_epoch = np.sum(contributions_per_epoch, axis=0)
		# for organization_idx in range(organization_num):
		# 	# Normalize the contributions for each organization by dividing the contributions by the sum of contributions for each epoch
		# 	normalized_contributions = (contributions_per_epoch[organization_idx] - np.min(total_contributions_per_epoch)) / (np.max(total_contributions_per_epoch) - np.min(total_contributions_per_epoch))

		# 	ax.plot(normalized_contributions, label=f"Org {organization_idx}")
		# 	print('Organization', organization_idx, 'contribution:', torch.tensor(contributions_per_epoch[organization_idx]))
		
		# ax.legend()
		# ax.set_xlabel('Epoch')
		# ax.set_ylabel('Contribution')

		# # Save the plot as a PNG image file
		# plt.savefig(f'{args.contribution_schem}_{args.attack}.png')

		#print(org_contributions)
		#print('attributions across epochs:', attr_lst)
				   
					

				
 
	elif model_type == 'centralized':
		
		chy_one_hot_enc = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
		X = chy_one_hot_enc.fit_transform( X )
			
	
		print('Client data shape: {}, postive ratio: {}'.format(X.shape, sum(y)/len(y)))
		
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	
		X_train = torch.from_numpy(X_train).float()
		X_test = torch.from_numpy(X_test).float()
		# y_train = torch.from_numpy(y_train).long()
		# y_test = torch.from_numpy(y_test).long()

		y_train = torch.from_numpy(y_train).float()
		y_test = torch.from_numpy(y_test).float()
	
		train_data = TensorDataset(X_train, y_train)
		test_data = TensorDataset(X_test, y_test)
	
		train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
		test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
		# sm = SMOTE(sampling_strategy='minority', k_neighbors=10, n_jobs=8 )
		# X_train, y_train = sm.fit_resample( X_train, y_train )
		# print('SMOTE, X_train.shape: ', X_train.shape)
		
		hidden_units = np.array([128,64,32])
	
		model = MlpModel(input_dim=X_train.shape[-1], hidden_units=hidden_units, num_classes=1)
		optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
		
		
		# print(X_train[0].shape[0])
		# flops, params = profile(model(), input=(X_train[0],0))
		# print('FLOPs = ' + str(flops/1000**3) + 'G')
		# print('Params = ' + str(params/1000**2) + 'M')
		
		# criterion = nn.CrossEntropyLoss()
		criterion = nn.BCELoss()
	
		model.train()
		for i in range(epochs):
			for idx, (data, targets) in enumerate(train_loader):
				optimizer.zero_grad()
	
				# compute output
				outputs = model(data)
				logits = torch.sigmoid(outputs)
				logits = torch.reshape(logits, shape = [len(logits)])
				# print(outputs.dtype, targets.dtype)
				loss = criterion(logits, targets)
				loss.backward()
				optimizer.step()
	
			for idx, (data, targets) in enumerate(test_loader):
				outputs = model(data)
				log_probs = torch.sigmoid(outputs)
				# y_pred = np.argmax(log_probs.data, axis=1)
				# acc = accuracy_score(y_true=targets.data, y_pred=y_pred)
				auc = roc_auc_score(targets.data, log_probs.data)
			print('For the {}-th epoch, test auc: {}'.format(i, auc))
			
			test_epoch_array.append(i+1)
			loss_array.append(loss.detach().numpy())
			auc_array.append(auc)
			
	return test_epoch_array, loss_array, auc_array

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='vertical FL')
	parser.add_argument('--dname', default='ADULT', help='dataset name: AVAZU, ADULT')
	parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')  
	parser.add_argument('--batch_type', type=str, default='mini-batch')
	parser.add_argument('--batch_size', type=int, default=500)
	parser.add_argument('--data_type', default='original', help='define the data options: original or one-hot encoded')
	parser.add_argument('--model_type', default='centralized', help='define the learning methods: vrtical or centralized')    
	parser.add_argument('--organization_num', type=int, default='2', help='number of origanizations, if we use vertical FL')
	parser.add_argument('--contribution_schem',  type=str, default='ig',help='define the contribution evaluation method')
	parser.add_argument('--attack', default='original', help='define the data attack or not')
	args = parser.parse_args()

	start_time = time.time()
	
	# collect experimental results
	test_epoch_array, loss_array, auc_array = main(args)

	elapsed = time.time() - start_time
	mins, sec = divmod(elapsed, 60)
	hrs, mins = divmod(mins, 60)
	print("Elapsed time: %d:%02d:%02d" % (hrs, mins, sec))
	print(test_epoch_array)
	print(loss_array)
	print(auc_array)
	# quickly plot figures to see the loss and acc performance
	# figure_loss = plt.figure(figsize = (8, 6))
	# plt.plot(test_epoch_array, loss_array)
	# plt.xlabel('Communication round #')
	# plt.ylabel('Test loss')
	# plt.ylim([0, 1])
	# 
	# figure_acc = plt.figure(figsize = (8, 6))
	# plt.plot(test_epoch_array, auc_array)
	# plt.xlabel('Communication round #')
	# plt.ylabel('Test accuracy')
	# plt.ylim([0, 1])  
# 