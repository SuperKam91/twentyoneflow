import numpy as np
import sklearn.externals.joblib

def get_data(z_f = '../data/zData_090219.txt', p_f = '../data/PT_v4_test.txt', s_f = '../data/T21data_v4_test.txt'):
	"""
	re-parameterise data so that for signal S, cosmo parameters {p}, redshift z: S_z({p}) -> S({p}, z)
	order of parameters in params file SHOULD BE:
	f_{star}, V_c, f_X, slope of X-ray SED (alpha), nu_min of X-ray SED, tau, R_mfp, zeta
	z_f is redshift file, p_f is cosmo param file, s_f is signal file
	"""
	z = np.genfromtxt(z_f) #uses same redshift binning as first datasets
	n_z = len(z) #136
	par = np.genfromtxt(p_f)
	phys_bool_a = par[:, -1] / par[:, 0] < 40000. #physical values according to cohen
	par = par[phys_bool_a, :] #physical
	pars = np.repeat(par, n_z, axis=0)
	twenty_one = np.genfromtxt(s_f)
	twenty_one = twenty_one[phys_bool_a, :] #physical
	twenty_ones = twenty_one.reshape(-1, 1)
	zs = np.stack([z for _ in range(twenty_one.shape[0])], axis=0).reshape(-1, 1)
	full = np.hstack([pars, zs, twenty_ones]) #columns are {p}, z, S
	x = full[:,:-1]
	y = full[:,-1].reshape(-1,1)     
	return x, y   

def scale_data(x, y, x_scale_f = '../saved_models/scalers/9_params_21_2_x_scaler.pkl', y_scale_f = '../saved_models/scalers/9_params_21_2_y_scaler.pkl', par_slice = range(7) + range(8,9)):
	"""
	scale inputs/output of model using same transformation used in training.
	Files should contain serialised sklearn scaler objects
	par_slice denotes subset of cosmological parameters used in model. By default
	model ignores zeta as input, as cohen's paper also didn't use it
	"""
	x_scaler = sklearn.externals.joblib.load(x_scale_f)
	y_scaler = sklearn.externals.joblib.load(y_scale_f)
	x_scaler.transform(x)
	y_scaler.transform(y)
	x = x[:,par_slice] 
	return x, y, x_scaler, y_scaler