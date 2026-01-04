from lifelines.statistics import logrank_test
from joblib import Parallel, delayed

import functionF as f
import pandas as pd
import numpy as np
import datetime
import random
import time
import os


def task_train( list_path_files, groups, size_min, range_nz_wg, num_max_inds, p_survivors, 
                 epoch, prob_parent, prob_mutation, prob_crossover, num_model, n_jobs,
                 stop_event, log_queue,
                 
          ):
    try:
        task_train2( list_path_files, groups, size_min, range_nz_wg, num_max_inds, p_survivors, 
                     epoch, prob_parent, prob_mutation, prob_crossover, num_model, n_jobs,
                     stop_event, log_queue,  
                   )   
    except Exception as e:
        log_queue.put(f"错误 {e}\n")  

def task_train2( list_path_files, groups, size_min, range_nz_wg, num_max_inds, p_survivors, 
                 epoch, prob_parent, prob_mutation, prob_crossover, num_model, n_jobs,
                 stop_event, log_queue,
          ):
    print = log_queue.put
    range_nz_wg = [1, range_nz_wg ]
    num_survivors = int( p_survivors/100 * num_max_inds )
    prob_parent = prob_parent / 100
    is_keep_survivors = False
    prob_mutation_weight_gene = prob_mutation_decision_gene = prob_mutation/100
    prob_crossover = prob_crossover/100
    
    print( f'> groups = {groups}\n')
    print( f'> size_min = {size_min}\n')
    print( f'> range_nz_wg = {range_nz_wg}\n')
    print( f'> num_max_inds = {num_max_inds}\n')
    print( f'> num_survivors = {num_survivors}\n')
    print( f'> epoch = {epoch}\n')
    print( f'> prob_parent = {prob_parent}\n')
    print( f'> prob_mutation = {prob_mutation_weight_gene}\n')
    print( f'> prob_crossover = {prob_crossover}\n')
    print( f'> num_model = {num_model}\n')
    print( f'> n_jobs = {n_jobs}\n')
    
    ## Load data
    list_data_cohort = []
    for path_file in list_path_files:
        list_data_cohort.append( f.read_data( path_file ).to_numpy() )
    
    
    o_shape = None
    for index, data_cohort in enumerate( list_data_cohort ):
        print( f'> file_cohort{index+1} : shape={data_cohort.shape}, path={list_path_files[index]}\n' )
        if o_shape == None:
            o_shape = data_cohort.shape
        else:
            if o_shape[1] != data_cohort.shape[1]:
                raise Exception( 'shape != o_shape' )
    
    path_dir = os.path.dirname( list_path_files[0] )
    
    num_data_cohort1, num_genes_weight = o_shape
    num_genes_weight   = ( num_genes_weight - 4 )
    num_genes_decision = ( groups - 1 )
    num_genes = ( num_genes_weight + num_genes_decision )
    
    #
    population = np.zeros([0,num_genes])
    chromosomes_loader = f.loader( num_genes )
    t1 = time.time()
    survivors = np.zeros([0,num_genes])
    list_fitness_survivors = np.zeros([0,])
    
    print( '> \n 开始构建 \n\n')
    for iter_ in range( epoch+1 ):        
        now_time = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
        
        
        ## ------ 1. Add individuals to the population ------
        population, num_miss = f.add_inds_to_pop(   population = population, 
                                                  num_max_inds = num_max_inds,
                                                   range_nz_wg = range_nz_wg,
                                              num_genes_weight = num_genes_weight,
                                                num_data_train = num_data_cohort1,
                                               )
        # population = np.unique( population, axis=0 )
        
        ## ------ 2.Evaluate individual fitness ------    
        list_num_viaSingleCohort = []
        for index, data_cohort in enumerate( list_data_cohort ):
            
            #
            if index == 0:
                [ list_fitness_cohort1, list_p_cohort1, list_values_decison
                ] = f.calculate_individuals_fitness(      population = population, 
                                                                data = data_cohort, 
                                                  num_genes_decision = num_genes_decision, 
                                                            is_train = True,
                                                            size_min = size_min,
                                                              n_jobs = n_jobs,
                                                   )
                list_num_viaSingleCohort.append( (list_p_cohort1 < 0.05 ).sum().item() )
            else:            
                population_temp = np.copy( population )
                population_temp[:,-num_genes_decision:] = list_values_decison
                #
                [ list_fitness_cohort1_2, list_p_cohort1_2, _
                ] = f.calculate_individuals_fitness(      population = population_temp, 
                                                                data = data_cohort, 
                                                  num_genes_decision = num_genes_decision, 
                                                            is_train = False,
                                                            size_min = size_min,
                                                              n_jobs = n_jobs,
                                                    ) 
                #
                list_fitness_cohort1 = np.minimum( list_fitness_cohort1, list_fitness_cohort1_2)
                list_p_cohort1 = np.maximum( list_p_cohort1, list_p_cohort1_2 )
                #
                list_num_viaSingleCohort.append( (list_p_cohort1_2 < 0.05 ).sum().item() )
        
        ## ------ 3.Detector ------
        loc_good = np.where( list_p_cohort1 < 0.05  )
        chromosomes = population[ loc_good ]

        if loc_good[0].size > 0 :
            chromosomes[:,-num_genes_decision:] = list_values_decison[ loc_good ]

                       
        ## ------ 4.Loaderrrr ------
        chromosomes_loader.load( chromosomes, is_unique=True )

        ## Print information
        t2 = time.time() - t1
        print( f'> Process:{iter_}/{epoch}, '+
               f'Num-viaSingle:{list_num_viaSingleCohort}, '+
               f'Num-viaAll:{chromosomes_loader.num()}, '+
               f'time:{t2:.2f}s     \n',
             )                   
        t1 = time.time()
            
        ## ------ 5.Terminator ------
        if (iter_ >= epoch) or (chromosomes_loader.num() >= num_model ) or stop_event.is_set():
            if stop_event.is_set():
                print("手动中止,保存结果\n")
            path_ = f'{path_dir}/chroms_g{groups}_r{range_nz_wg[-1]}_t({now_time}).csv'
            print(f"保存到 {path_}\n")
            chromosomes_loader.save_chromosomes( path_, groups )
            print( '\n> Execution Completed \n')
            break
    
        ## ------ 6. Generate offspring of the population ------
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
        # print( f'{population[:,:-num_genes_decision].max()}，{population[:,:-num_genes_decision].min()}\n')
 
    
##_____________________________________________________________________________

def task_infer( list_path_files, model_path, log_queue, ):
    print = log_queue.put
    chromosome = pd.read_csv( model_path, header=None).to_numpy()
    if len( chromosome ) > 1:
        print( '模型文件 里只能有 一个 模型\n' )
        return
    num_genes_decision = int( chromosome[:,-1].item() ) -1
    chromosome = chromosome[:,:-1] 
    
    for index, path_file in enumerate( list_path_files ):   
        #
        file = f.read_data( path_file )
        data_cohort = file.to_numpy()
        #
        [ list_score, _, list_interval_groups, _, _,
        ] = f.get_decision_information(       population = chromosome, 
                                                    data = data_cohort, 
                                      num_genes_decision = num_genes_decision, 
                                                is_train = False,
                                    )
        list_score = list_score.T
        
        list_locs_group = []
        for ig in list_interval_groups[0]: 
            locs = np.where( np.greater( ig[0], list_score ) & \
                             np.greater_equal( list_score, ig[1] ) 
                           )[0]
            list_locs_group.append( locs )
    
        ##
        list_score_col = list_score.reshape(-1, 1)
    
        n_samples = data_cohort.shape[0]
        group_col = np.zeros((n_samples, 1), dtype=int)
    
        for gid, locs in enumerate(list_locs_group, start=1):
            group_col[locs, 0] = gid
    
    
        ## save
        path_dir = os.path.dirname( path_file )
        basename = os.path.basename( path_file )
        first_col = file.iloc[:, 0]
        result_df = pd.DataFrame({
            first_col.name: first_col.values,              
            "pre_score": list_score_col.flatten(),          
        })
    
        result_df["group"] = group_col.flatten()
        
        path_save = f'{path_dir}/pre_{basename}'
        print( f'> 保存结果 {path_save} \n')
        result_df.to_csv( path_save, index=False )
                


    
# if __name__ == "__main__":
    
#     list_path_files = [
#     'NPC_cohort1_preprocessed.csv',
#     'NPC_cohort2_preprocessed.csv',    
#         ]
    
#     groups = 2 # 2 or 2
#     size_min = 10 # The minimum number of samples
    
#     #
    
#     range_nz_wg = 10 # The range of the number of non-zero weight genes
#     num_max_inds = 500 # The maximum number of individuals in the population
#     p_survivors = 80 # The number of survivors
#     epoch = 10000 # The number of epochs
#     prob_parent = 90 # The probability of selecting the parent

#     prob_mutation = 90
#     prob_crossover = 50 # The probability of crossover
    
#     n_jobs = 16 # 
#     num_model = 5
    
#     task_train2( list_path_files, groups, size_min, range_nz_wg, num_max_inds, p_survivors, 
#                   epoch, prob_parent, prob_mutation, prob_crossover, num_model, n_jobs,
#           )


# if __name__ == "__main__":
    
#     list_path_files = [
#     'D:/Desktop/tkinter_test/NPC_cohort2_preprocessed.csv',  
#         ]
    
#     model_path = 'D:/Desktop/tkinter_test/chroms_g2_r10_t(2026_0103_2229) - 1221.csv'
    
#     task_infer( list_path_files, model_path )
    
    
    
    
    
    
    
    
    
    
    
    
    