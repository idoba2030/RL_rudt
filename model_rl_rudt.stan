functions {

  real log_likelihood_lpdf(vector rt_ndt, vector P, vector P_C,vector theta) {

    vector [num_elements(rt_ndt)] prob;
    vector [num_elements(rt_ndt)] sqrt_P_C=sqrt(P_C);
    for (i in 1:num_elements(rt_ndt)){
  
  // Now use sum_value in your model, e.g.:
  prob[i] =(P[i] / (theta[i] * sqrt_P_C[i])) * exp(-rt_ndt[i] / theta[i]) * (exp(rt_ndt[i] * sqrt_P_C[i] / theta[i])-cosh(rt_ndt[i] * sqrt_P_C[i] / theta[i]));

    }
    return sum(log(prob));

  }

}

data {
   int<lower=1> Ndata;                // Total number of trials (for all subjects)
   int<lower=1> Nsubjects;            // Number of subjects
   int<lower=1> Narms;                // Number of arms
   int<lower=1> Nraffle;              // Number of offered arms
   int<lower=1> Ndims;                // Number of feature dimensions
   
  array [Ndata] int <lower=1, upper=Nsubjects> subject_trial; // Which subject performed each trial
 

  array [Ndata] int <lower=1,upper=4>card_left; // 1 to 4
  array [Ndata] int <lower=1,upper=4>card_right; // 1 to 4
  array [Ndata] int <lower=1,upper=4>ch_card; // Ch_card 1 to 4
  array [Ndata] int <lower=1,upper=2>ch_key; // Ch_key 1 for left, 2 for right
  array [Ndata] int <lower=0,upper=1>reward; // reward
  array [Ndata] int <lower=0,upper=1>first_trial_in_block; // binary indicator
  array [Ndata] real Qnet_diff;
  
  array [Ndata] real rt;                  // Response times for all trials
  vector <lower=0> [Nsubjects] min_rt; //minimal response time for every subject to block tau.

}


parameters {
  // Group-level (population) parameters
  real mu_alpha;        // Mean learning rate across subjects
  real mu_lambda;        // Mean threshold across subjects
  real mu_c;            // Mean scaling_cdf across subjects
  real mu_theta;        // Mean scaling_gamma across subjects
  real mu_tau;          // Mean non-decision time across subjects

  // Group-level standard deviations (for subject-level variability)
  real<lower=0> sigma_alpha;       // Variability in learning rate
  real<lower=0> sigma_lambda;       // Variability in threshold
  real<lower=0> sigma_c;            // Variability in scaling_cdf 
  real<lower=0> sigma_theta;        // Variability in scaling gamma
  real<lower=0> sigma_tau;          // Variability in non-decision time
  
// Non-centered parameters (random effects in standard normal space)
  vector[Nsubjects] alpha_raw;          // Standard normal random effects for learning rate
  vector[Nsubjects] lambda_raw;          // Standard normal random effects for threshold
  vector[Nsubjects] c_raw;           // Standard normal random effects for scaling_cdf 
  vector[Nsubjects] theta_raw;               // Standard normal random effects for scaling gamma
  vector[Nsubjects] tau_raw;             // Standard normal random effects for non-decision time
}
transformed parameters {
  vector<lower=0>[Nsubjects] alpha_sbj; // learning rate
  vector<lower=0>[Nsubjects] lambda_sbj; // Threshold
  vector<lower=0>[Nsubjects] c_sbj; // Scaling of sigma of CDF
  vector<lower=0>[Nsubjects] theta_sbj; // Scale of gamma distribution
  vector<lower=0>[Nsubjects] tau_sbj; // Non-decision time
  
  // Non-centered parameterization: scaling the raw variables
  for (subject in 1:Nsubjects){
 alpha_sbj[subject] = inv_logit(mu_alpha +sigma_alpha* alpha_raw[subject]);
 lambda_sbj[subject] = log1p_exp(mu_lambda +sigma_lambda* lambda_raw[subject]);
 c_sbj[subject] = log1p_exp(mu_c + sigma_c * c_raw[subject]);
 theta_sbj[subject] = log1p_exp(mu_theta +sigma_theta * theta_raw[subject]);
 tau_sbj[subject] = log1p_exp(mu_tau + sigma_tau * tau_raw[subject]);
  }

	vector [Ndata]lambda_t;			// trial-by-trial treshold
	vector [Ndata]c_t;					  // trial-by-trial scaling cdf
	vector [Ndata]theta_t;       // trial-by-trial scaling gamma
	vector[Ndata]tau_t;         // trial-by-trial non_decision
	real Qnet_diff;
	
	vector <lower=0>[Ndata] sigma;
	vector <lower=0>[Ndata] P_L;
	vector <lower=0>[Ndata] P_S;
	vector <lower=0>[Ndata] P_C;
	vector [Ndata] P;
	vector [Ndata] rt_ndt;
	
	for (t in 1:Ndata){
	  lambda_t[t] = lambda_sbj[subject_trial[t]];
		c_t[t] = c_sbj[subject_trial[t]];
		theta_t[t] = theta_sbj[subject_trial[t]];
		tau_t[t] = tau_sbj[subject_trial[t]];
	  
	  Qnet_diff=Qnet_diff[t];

    sigma[t] = sqrt(c_t[t] * abs(Qnet_diff)); // Added a small constant to avoid zero;

    P_L[t] = 1 - normal_cdf(lambda_t[t] - Qnet_diff | 0, sigma[t]);

    P_S[t] = normal_cdf(-lambda_t[t]- Qnet_diff | 0, sigma[t]);
    
    P_C[t] = 1 - P_S[t] - P_L[t];
    
    
    P[t] = choice[t] == 1 ? P_S[t] : P_L[t];

    rt_ndt[t] = fmax(rt[t] - tau_t[t],1e-8);
	}
}
model {
  // Priors for group-level parameters
  mu_lambda ~ normal(-1, 2);
  mu_c ~ normal(0.5, 2);
  mu_theta ~ normal(-1, 2);
  mu_tau ~ normal(-1, 2);
  
  // Priors for group-level standard deviations
  sigma_lambda ~ normal(0, 1);
  sigma_c ~ normal(0, 1);
  sigma_theta ~ normal(0, 1);
  sigma_tau ~ normal(0, 1);
  
  // Priors for subject-specific effect
  lambda_raw~normal(0,1);
  c_raw~normal(0,1);
  theta_raw~normal(0,1);
  tau_raw~normal(0,1);
  
rt_ndt~log_likelihood( P,
                            P_C,
                            theta_t);
    }
