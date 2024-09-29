library(dplyr)
#### True values ----
Ntrials = 100

# Free parameters
alpha             = 0.3    # learning rate     
c                 = 0.8    # scaling for sigma of the value difference normal dist
lambda            = 0.3    # threshold for sampler
theta             = 0.3    # scale parameter for gamma
non_decision_time = 0.2

#### Generate decision ----

rts     = rep(0, Ntrials)
choices = rep(0, Ntrials)
ds = rep(0, Ntrials)
reward_probs=c(0.3,0.8)

Qvalues= rep(0.5, 2)
for (trial in 1:Ntrials) {
  d       = Qvalues[2] - Qvalues[1]
  sigma   = sqrt(c * abs(d)) +0.3 #0.3 to make it run.
  rt = 0
  x  = 0
  while (abs(x) < lambda) {
    rt = rt + rgamma(1, shape = 2, scale = theta)
    x  = rnorm(1, d, sigma)
  }
  
  ch_card = if_else(sign(x)==1,2,1)
  rt     = rt + non_decision_time
  reward = sample(0:1, 1, prob = c(1-reward_probs[ch_card],reward_probs[ch_card])) #reward according to card
  choices[trial] = chosen_card
  rts[trial]     = rt
  ds[trial]=d
  PE=reward-Qvalues[ch_card]
  Qvalues[ch_card] = Qvalues[ch_card]  + alpha * PE
}
