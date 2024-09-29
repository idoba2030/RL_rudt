#### simulate Rescorla-Wagner block for participant ----
sim.block = function(subject,parameters,cfg){ 
  print(paste('subject',subject))
  
  #pre-allocation
  #set parameters
  alpha = parameters['alpha']
  lambda = parameters['lambda'] #threshold
  c  = parameters['c'] #scaling of sigma
  theta = parameters['theta'] #scale of gamma dist
  tau  = parameters['tau'] #non decision time

  #set initial var
  Nblocks            = cfg$Nblocks
  Ntrials            = cfg$Ntrials
  Ndims              = cfg$Ndims
  Narms              = cfg$Narms
  Nraffle            = cfg$Nraffle
  expvalues          = cfg$rndwlk
  df                 =data.frame()
  
  prior_relevant=c(1,0) #instructions
  weight_uniform=rep(1/Ndims,Ndims) #no instructions
  weights= 1* prior_relevant + 0 * weight_uniform
  
  for (block in 1:Nblocks){
    Q_cards= rep(0.5, Narms)
    Q_keys = rep(0.5, Nraffle)
    for (trial in 1:Ntrials){
      
      options=sample(1:4,2)
      Q_cards_offered = Q_cards[options] #use their Q values
      
      Qnet = weights[1]*Q_cards_offered + weights[2]*Q_keys
      
      Qnet_diff=Qnet[2]-Qnet[1]

      sigma    = sqrt(c * abs(Qnet_diff))
      rt = 0
      x  = 0
      while (abs(x) < lambda) {
        rt = rt + rgamma(1, shape = 2, scale = theta)
        x  = rnorm(1, mean=Qnet_diff, sd=sigma)
      }
      rt     = rt + tau
      ch_key = if_else(sign(x)==1,2,1)
      ch_card=options[ch_key]
      #outcome 
      reward = sample(0:1, 1, prob = c(1 - expvalues[ch_card, trial], expvalues[ch_card, trial])) #reward according to card
      
      #calculate PEs
      PE_keys= reward-Q_keys[ch_key]
      PE_cards=reward-Q_cards[ch_card]
      
      #create data for current trials
      dfnew=data.frame(subject=subject,
                       block=block,
                       trial = trial,
                       first_trial_in_block=if_else(trial==1,1,0),
                       rt = rt,        
                       card_left=options[1],
                       card_right=options[2],
                       ch_key,
                       ch_card,
                       Qnet_diff
                       )

      
      Q_cards[ch_card] = Q_cards[ch_card]  + alpha * PE_cards
      Q_keys[ch_key] = Q_keys[ch_key] +alpha * PE_keys
      df=rbind(df,dfnew)

    }
  }     
  return (df)
}