from lifelines.statistics import logrank_test
from joblib import Parallel, delayed
import pandas as pd

import function as f
import datetime
import numpy as np
import random
import time
import os

## Parameters _________________________________________________________________
path_train_file = './file_1.csv'
path_valid_file = './file_2.csv'

groups = 3
size_min = 10

n_jobs = 40

range_nz_wg = [1,8] # The range of the number of non-zero weight genes
num_max_inds = 500
num_survivors = 350
epoch = 10000
prob_parent = 0.9
prob_mutation_weight_gene = 0.9
prob_mutation_decision_gene = 0.9
prob_crossover = 0.5
num_iter_warmup = 0
is_keep_survivors = False

# np.random.seed(2023)
path_pre_population = None


## Main _______________________________________________________________________
if groups == 2:
    num_iter_warmup = -1
#
## Load data
data_train = f.read_data( path_train_file ).to_numpy()
data_valid = f.read_data( path_valid_file ).to_numpy()
# array_features_train = data_train[:,4:]
num_data_train, num_genes_weight = data_train.shape
num_genes_weight   = ( num_genes_weight - 4 )
num_genes_decision = ( groups - 1 )
num_genes = ( num_genes_weight + num_genes_decision )
#

print( f'> file_train : shape={data_train.shape}, path={path_train_file}' )
print( f'> file_valid : shape={data_valid.shape}, path={path_valid_file}' )
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
    population = np.zeros( [ 0, num_genes ] )
else:
    pre_population = np.load( path_pre_population )
    population, num_miss = f.add_inds_to_pop(   
                                         population = np.zeros([ 0,num_genes]), 
                                       num_max_inds = len( pre_population ),
                                        range_nz_wg = range_nz_wg,
                                   num_genes_weight = num_genes_weight,
                                     num_data_train = num_data_train,
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
    population, num_miss = f.add_inds_to_pop(     population = population, 
                                                num_max_inds = num_max_inds,
                                                 range_nz_wg = range_nz_wg,
                                            num_genes_weight = num_genes_weight,
                                              num_data_train = num_data_train,
                                            )
    # population = np.unique( population, axis=0 )
    
    ## ------ 2.Evaluate individual fitness ------
    is_warmup = False
    [ list_fitness_train, list_p_train, list_values_decison 
    ] = f.calculate_individuals_fitness(        population = population, 
                                                      data = data_train, 
                                        num_genes_decision = num_genes_decision, 
                                                  is_train = True,
                                                  size_min = size_min,
                                                    n_jobs = n_jobs,
                                                 is_warmup = is_warmup,
                                       )

    ## ------ 3.Detector ------
    [ chromosomes, p_chromosomes, num_arcoss_train,
    ] = f.detector(                data = data_valid, 
                             population = population, 
                    list_values_decison = list_values_decison, 
                                 list_p = list_p_train, 
                               size_min = size_min,
                                 n_jobs = n_jobs,
                              is_warmup = is_warmup,
                  )

    ## ------ 4.Loader ------
    #
    chromosomes_loader.load( chromosomes, is_unique=True )
    # temp saver
    if iter_ % 10 == 0:
        chromosomes_loader.save_chromosomes( f'chroms_g{groups}_r{range_nz_wg[-1]}_temp_f({path_train_file}).npy' )
        # 
        if (not is_warmup) and ( chromosomes_loader.num() < 5 ) and groups == 3:
            [ chromosomes_notgood, _, num_arcoss_train_notgood,
            ] = f.detector(         data = data_valid, 
                              population = population, 
                      list_values_decison = list_values_decison, 
                                  list_p = list_p_train, 
                                size_min = size_min,
                                  n_jobs = n_jobs,
                               is_warmup = True,
                          )

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
        chromosomes_loader.save_chromosomes( f'chroms_g{groups}_r{range_nz_wg[-1]}_f({path_train_file})_t({now_time}).npy' )
        if groups == 2:
            chromosomes_notgood_loader.save_chromosomes( f'chroms_notgood_g{groups}_r{range_nz_wg[-1]}_f({path_train_file})_t({now_time}).npy' )
        print( '\n> Execution Completed ')
        break


    ## ------ 6. Generate offspring of the population ------
    # Find strong survivors --------------------------------------------------
    if is_keep_survivors:
      locs = np.argsort( list_fitness_survivors )[::-1] # Sort from large to small
      survivors = survivors[ locs ]
      list_fitness_survivors = list_fitness_survivors[ locs ]
      
      _, locs = np.unique( ( survivors[:,:num_genes_weight] != 0 ), 
                            axis = 0,
                            return_index = True,
                          ) # Remove Duplicates
      #
      locs = locs[:2]
      population = np.concatenate( [ survivors[ locs ], population ], 
                                  axis=0 
                                )
      list_fitness_train = np.concatenate( [ list_fitness_survivors[ locs ], 
                                            list_fitness_train,
                                          ], 
                                          axis=0 
                                        )
    
    # Selection of survivors based on fitness ----------------------------------
    survivors, list_fitness_survivors = f.select_survivors( 
                                              population = population, 
                                      list_fitness_train = list_fitness_train, 
                                           num_survivors = num_survivors,
                                                           )
    # survivors = np.concatenate( [ top, survivors ], axis=0 )
    
    # -- Selection parents --
    indices_parents_2 = f.get_indices_parents( population  = survivors, 
                                               prob_parent = prob_parent,
                                           )
    # -- Parents crossover --
    offspring = f.crossover(        population = survivors, 
                             indices_parents_2 = indices_parents_2, 
                                 prob_crossover = prob_crossover,
                           )
    # -- offspring mutation --
    offspring_m = f.mutation(         population = offspring, 
                                num_genes_weight = num_genes_weight, 
                       prob_mutation_weight_gene = prob_mutation_weight_gene, 
                     prob_mutation_decision_gene = prob_mutation_decision_gene,
                                  num_data_train = num_data_train,
                           )
    offspring_m = f.controller( offspring_m, range_nz_wg, num_genes_decision )
    
    # 
    # -- Updata population --
    population = offspring_m
    # print( population[:,:-num_genes_decision].max() )


