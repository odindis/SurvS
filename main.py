from lifelines.statistics import logrank_test
from joblib import Parallel, delayed

import functionF as f
import pandas as pd
import numpy as np
import datetime
import random
import time
import os

## Parameters _________________________________________________________________
# Input xlsx or csv
# The first column is the serial number,
# The second column is duration
# The third column is the event
# The fourth column is measure
# The fifth to last column are the features.
# The first column has no restrictions and can be arranged freely

path_cohort1_file = 'NPC_cohort1_preprocessed.xlsx'
path_cohort2_file = 'NPC_cohort2_preprocessed.xlsx'



groups = 2 # 2 or 2
size_min = 10 # The minimum number of samples

#
n_jobs = 32 # 

range_nz_wg = [1,10] # The range of the number of non-zero weight genes
num_max_inds = 500 # The maximum number of individuals in the population
num_survivors = 400 # The number of survivors
epoch = 10000 # The number of epochs
prob_parent = 0.9 # The probability of selecting the parent
prob_mutation_weight_gene = 0.9 # The probability of mutation
prob_mutation_decision_gene = 0.9 # The probability of mutation
prob_crossover = 0.5 # The probability of crossover
num_iter_warmup = 50 # The number of iterations for warmup
is_keep_survivors = False # keep storing individuals

# np.random.seed(2023)
path_pre_population = None #

## Main _______________________________________________________________________
if groups == 2:
    num_iter_warmup = -1
#
## Load data
data_cohort1 = f.read_data( path_cohort1_file ).to_numpy()
data_cohort2 = f.read_data( path_cohort2_file ).to_numpy()

num_data_cohort1, num_genes_weight = data_cohort1.shape
num_genes_weight   = ( num_genes_weight - 4 )
num_genes_decision = ( groups - 1 )
num_genes = ( num_genes_weight + num_genes_decision )
#

print( f'> file_cohort1 : shape={data_cohort1.shape}, path={path_cohort1_file}' )
print( f'> file_cohort2 : shape={data_cohort2.shape}, path={path_cohort2_file}' )
print( f'> groups = {groups}')
print( f'> range_nz_wg = {range_nz_wg}')
print( f'> size_min = {size_min}')
print( f'> n_jobs = {n_jobs}')
print( f'> num_iter_warmup = {num_iter_warmup}')
print( f'> num_genes_weight = {num_genes_weight}')
print( f'> num_genes_decision = {num_genes_decision}')
print()

#
if path_pre_population == None:
    population = np.zeros([0,num_genes])
else:
    pre_population = np.load( path_pre_population )
    population, num_miss = f.add_inds_to_pop(   
                                         population = np.zeros([ 0,num_genes]), 
                                       num_max_inds = len( pre_population ),
                                        range_nz_wg = range_nz_wg,
                                   num_genes_weight = num_genes_weight,
                                     num_data_train = num_data_cohort1,
                                             )
    population[:,:-num_genes_decision] = pre_population[:,:-num_genes_decision]
#   
chromosomes_loader = f.loader( num_genes )
chromosomes_loader_warmup = f.loader( num_genes )
chromosomes_notgood_loader = f.loader( num_genes )
t1 = time.time()
survivors = np.zeros([0,num_genes])
list_fitness_survivors = np.zeros([0,])

for iter_ in range( epoch+1 ):        
    now_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
    ## ------ 1. Add individuals to the population ------
    t_temp = time.time()
    population, num_miss = f.add_inds_to_pop(   population = population, 
                                              num_max_inds = num_max_inds,
                                               range_nz_wg = range_nz_wg,
                                          num_genes_weight = num_genes_weight,
                                            num_data_train = num_data_cohort1,
                                           )
    # population = np.unique( population, axis=0 )
    
    ## ------ 2.Evaluate individual fitness ------
    t_temp = time.time()
    is_warmup = True if iter_ < num_iter_warmup else False
    
    
    [ list_fitness_cohort1, list_p_cohort1, list_values_decison 
    ] = f.calculate_individuals_fitness(      population = population, 
                                                    data = data_cohort1, 
                                      num_genes_decision = num_genes_decision, 
                                                is_train = True,
                                                size_min = size_min,
                                                  n_jobs = n_jobs,
                                               is_warmup = is_warmup,
                                       )

    ## ------ 3.Detector ------
    t_temp = time.time()
    [ chromosomes, p_chromosomes, num_arcoss_train,
    ] = f.detector(          data = data_cohort2, 
                       population = population, 
              list_values_decison = list_values_decison, 
                           list_p = list_p_cohort1, 
                         size_min = size_min,
                           n_jobs = n_jobs,
                        is_warmup = is_warmup,
                  )

    ## ------ 4.Loader ------
    t_temp = time.time()
    # warmup
    if is_warmup:
        chromosomes_loader_warmup.load( chromosomes, is_unique=True )
        print( f'> Num-warmup-result:{chromosomes_loader_warmup.num()} ')
        chromosomes = np.zeros([0,num_genes])
    if iter_ == num_iter_warmup:
        chromosomes_loader_warmup.save_chromosomes( f'chroms_g{groups}_r{range_nz_wg[-1]}_warmup_f({path_cohort1_file})_t({now_time}).npy' )
        print('> Warmup Completed')
    #
    chromosomes_loader.load( chromosomes, is_unique=True )
    # temporary save
    if iter_ % 10 == 0:
        chromosomes_loader.save_chromosomes( f'chroms_g{groups}_r{range_nz_wg[-1]}_temp_f({path_cohort1_file}).npy' )
        # 装载过了训练集但没过验证集的
        if (not is_warmup) and ( chromosomes_loader.num() < 5 ) and groups == 3:
            [ chromosomes_notgood, _, num_arcoss_train_notgood,
            ] = f.detector(         data = data_cohort2, 
                              population = population, 
                      list_values_decison = list_values_decison, 
                                  list_p = list_p_cohort1, 
                                size_min = size_min,
                                  n_jobs = n_jobs,
                               is_warmup = True,
                          )
            chromosomes_notgood_loader.load( chromosomes_notgood, is_unique=True )
            if chromosomes_notgood_loader.num() > 0:
                chromosomes_notgood_loader.save_chromosomes( f'chroms_notgood_g{groups}_r{range_nz_wg[-1]}_temp_f({path_cohort1_file}).npy' )
            print( f'> Num-notgood-result:{chromosomes_notgood_loader.num()} ')
    
    ## Print information
    t2 = time.time() - t1
    print( f'> Process:{iter_}/{epoch}, '+
           f'Num-across-train:{num_arcoss_train}/{len(population)}, ' +
           f'Num-result:{chromosomes_loader.num()}, '+
           f'time-consum:{t2:.2f}s     ',
         )                   
    t1 = time.time()
        
    ## ------ 5.Terminator ------
    if (iter_ == epoch) or (chromosomes_loader.num() > 10000 ):
        chromosomes_loader.save_chromosomes( f'chroms_g{groups}_r{range_nz_wg[-1]}_f({path_cohort1_file})_t({now_time}).npy' )
        if groups == 2:
            chromosomes_notgood_loader.save_chromosomes( f'chroms_notgood_g{groups}_r{range_nz_wg[-1]}_f({path_cohort1_file})_t({now_time}).npy' )
        print( '\n> Execution Completed ')
        break



    ## ------ 6. Generate offspring of the population ------
    t_temp = time.time()
    # Find strong survivors --------------------------------------------------
    if is_keep_survivors:
      locs = np.argsort( list_fitness_survivors )[::-1] # sort, max to min
      survivors = survivors[ locs ]
      list_fitness_survivors = list_fitness_survivors[ locs ]
      
      _, locs = np.unique( ( survivors[:,:num_genes_weight] != 0 ), 
                            axis = 0,
                            return_index = True,
                          ) # 去重
      #
      locs = locs[:2]
      population = np.concatenate( [ survivors[ locs ], population ], 
                                  axis=0 
                                )
      list_fitness_cohort1 = np.concatenate( [ list_fitness_survivors[ locs ], 
                                            list_fitness_cohort1,
                                          ], 
                                          axis=0 
                                        )

    #--------------------------------------------------------------------------
    # Selection of survivors based on fitness
    survivors, list_fitness_survivors = f.select_survivors( 
                                              population = population, 
                                      list_fitness_train = list_fitness_cohort1, 
                                           num_survivors = num_survivors,
                                                           )
    # survivors = np.concatenate( [ top, survivors ], axis=0 )
    
    # -- Selection parents
    indices_parents_2 = f.get_indices_parents( population  = survivors, 
                                               prob_parent = prob_parent,
                                           )
    # -- Parents crossover 
    offspring = f.crossover(        population = survivors, 
                             indices_parents_2 = indices_parents_2, 
                                 prob_crossover = prob_crossover,
                           )
    # -- offspring mutation 
    offspring_m = f.mutation(         population = offspring, 
                                num_genes_weight = num_genes_weight, 
                       prob_mutation_weight_gene = prob_mutation_weight_gene, 
                     prob_mutation_decision_gene = prob_mutation_decision_gene,
                                  num_data_train = num_data_cohort1,
                           )
    offspring_m = f.controller( offspring_m, range_nz_wg, num_genes_decision )

    # 
    # -- Updata population
    population = offspring_m
    # print( population[:,:-num_genes_decision].max() )



