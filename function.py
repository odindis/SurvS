# Import Libraries ____________________________________________________________
from lifelines.statistics import logrank_test
from joblib import Parallel, delayed

import lifelines as ll
import pandas as pd
import numpy as np
import chardet

import warnings
warnings.filterwarnings('ignore')

## Main _______________________________________________________________________
def calculate_vif( x ):
    vif = pd.DataFrame()
    vif["Features"] = x.columns
    list_vif = []
    for i in range( x.shape[1] ):
        list_vif.append( smsoi.variance_inflation_factor( x.values, i ) )
    vif["VIF Factor"] = list_vif 
    return vif
#
def remove_highest_vif_feature( x_, non_target, measure_col ):
    x = x_.drop( non_target, axis=1 )
    vif = calculate_vif( x )
    vif = vif[ vif['Features'] != measure_col ]
    max_vif = vif['VIF Factor'].max()
    remove_feature = vif.loc[ vif['VIF Factor'] == max_vif, 'Features'].iloc[0]    
    # x = x.drop( remove_feature, axis=1 )
    x_ = x_.drop( remove_feature, axis=1 )
    return x_
#
def get_results_cox( cph, df, duration_col, event_col, measure_col, strata=None ):
    num = 0 
    while True:
        try:
            if num <= 5:
                cph.fit( df = df, 
                         duration_col = duration_col, 
                         event_col = event_col,
                         strata = strata,
                       )
            else:
                cph.penalizer = 0.1
                cph.fit( df = df, 
                         duration_col = duration_col, 
                         event_col = event_col,
                         strata = strata,
                       )
            break  
        except:
            num += 1
            non_target = [ duration_col, event_col ] 
            df = remove_highest_vif_feature( df, 
                                             non_target = non_target, 
                                             measure_col = measure_col,
                                           )
    return cph

def read_data( path_ ):

    with open( path_, 'rb' ) as f:
        encoding = chardet.detect( f.read() )['encoding']

    if path_.endswith( '.csv' ):
        data = pd.read_csv( path_, encoding=encoding )
    elif path_.endswith( '.xlsx' ):
        data = pd.read_excel( path_, encoding=encoding )
    else:
        raise ValueError( 'The file format is not supported!' )
    return data
#
def add_inds_to_pop( population, num_max_inds, range_nz_wg, 
                     num_genes_weight, num_data_train 
                   ):
    num_pop, num_genes = population.shape
    
    if num_pop >= num_max_inds:
        num_miss = 0
        return population, 0
    
    num_miss = ( num_max_inds - num_pop ) 
    pop_miss = np.zeros( [ num_miss, num_genes ] )
    #
    ## Weight genes part
    rnw = range_nz_wg
    num_nonzero = np.random.randint( rnw[0], rnw[1]+1, size = num_miss )
    # 
    indices_row = np.repeat( np.arange( num_miss ), num_nonzero )
    num_total = np.sum( num_nonzero )
    indices_col = np.random.choice( num_genes_weight, num_total, replace=True )
    
    # 
    values = np.random.uniform( -2, 2, size = num_total )
    # 
    pop_miss[:,:num_genes_weight][ indices_row, indices_col] = values
    #
    ## Decision genes part
    num_genes_decision = ( num_genes - num_genes_weight )
    r = int( num_data_train * 0.1 ) # 
    n = ( num_genes_decision + 1 )
    nrr = np.random.randint
    for i in range( 0, num_genes_decision ):
        rate = int( num_data_train * (i+1) / n ) # 
        pop_miss[:,num_genes_weight:][:, i] = nrr( rate-r, rate+r, 
                                                   size = num_miss 
                                                 )
    #
    population = np.concatenate( [population, pop_miss], axis=0 )
    population = np.round( population, 3 )
    return population, num_miss
#
#
def get_list_interval_groups( list_genes_decison, is_train ):
    #
    if is_train: # array position
        value_init = 0
        value_last = -1
    else: # value
        value_init =  np.inf
        value_last = -np.inf
    #
    list_i_groups = []
    #
    for genes_decison in list_genes_decison:
        temp = value_init
        interval_groups = []
        for gene_decison in genes_decison:
            interval_groups.append( [temp, gene_decison ] )
            temp = gene_decison
        interval_groups.append( [ gene_decison, value_last ] )  
        #
        list_i_groups.append( interval_groups )   
    #
    return list_i_groups
#
#
def get_decision_information( population, data, num_genes_decision, is_train ):

    list_score = np.dot( population[:, :-num_genes_decision], 
                         data[:,4:].T,
                        ) # decision values
    list_genes_decison = population[:, -num_genes_decision:] # weight
    if is_train:
        list_genes_decison = list_genes_decison.astype( np.int32 ) # position
    #
    glig = get_list_interval_groups
    list_interval_groups = np.array( glig( list_genes_decison, is_train ) )
        
    if is_train:
        # 
        list_indices_sorted = np.argsort( list_score, axis=1 )[:,::-1] 
        
        # position to value
        list_values_decison = []
        for i in range( len( list_score ) ):
            score = list_score[ i ]
            indices_sorted = list_indices_sorted[ i ]
            genes_decison = list_genes_decison[ i ]    
            values_decison = score[ indices_sorted ][ genes_decison ]
            list_values_decison.append( values_decison ) 
        list_values_decison = np.array( list_values_decison )
        
        list_positions_decison = list_genes_decison
        return list_score, list_indices_sorted, list_interval_groups, \
               list_positions_decison, list_values_decison
    else:
        list_values_decison = list_genes_decison
        return list_score, None, list_interval_groups, \
               None, list_values_decison
#
#
def get_event( data_group ):
    mask = ( data_group[ :, 3 ] == 0 )
    t0, e0 = data_group[  mask, 1 ], data_group[  mask, 2 ]
    t1, e1 = data_group[ ~mask, 1 ], data_group[ ~mask, 2 ]
    return t0, t1, e0, e1
#    

def calculate_difference( data, interval_groups, score, is_train, size_min, is_warmup=False ):
    # grouping and caculate p
    ig_first = interval_groups[0]
    ig_last  = interval_groups[-1]

    if is_train: #[ position_0 < position_0 ]
        data_group_first = data[ ig_first[0] : ig_first[1] ]     
        data_group_last  = data[ ig_last [0] : ig_last [1] ]
    else: #test # [ value_0 > value_1 ]
        data_group_first = data[ np.greater      ( ig_first[0], score ) & \
                                 np.greater_equal( score, ig_first[1] )
                               ]
        data_group_last  = data[ np.greater      ( ig_last[0], score ) & \
                                 np.greater_equal( score, ig_last[1] )
                               ]
    # 3 group ---
    if len( interval_groups ) == 3:
        # first
        t0, t1, e0, e1 = get_event( data_group_first )
        t0t = np.mean( t0 * (1-e0+1) )
        t1t = np.mean( t1 * (1-e1+1) )
        if ( len( t0 ) < size_min ) or ( len( t1 ) < size_min ) or ( t0t >= t1t ):
            p_first = 0.999
        else:
            p_first = logrank_test( t0, t1, e0, e1 ).p_value
        # last
        t0, t1, e0, e1 = get_event( data_group_last )
        if ( len( t0 ) < size_min ) or ( len( t1 ) < size_min ):
            p_last = 0.999
        else: 
            p_last = logrank_test( t0, t1, e0, e1 ).p_value #<===
        # return
        if is_warmup:
            return p_first
        else:
            return max( p_first, p_last )
    # 2 group ---
    elif len( interval_groups ) == 2: 
        # first
        t0, t1, e0, e1 = get_event( data_group_first )
        t0t = np.mean( t0 * (1-e0+1) )
        t1t = np.mean( t1 * (1-e1+1) )
        if ( len( t0 ) < size_min ) or ( len( t1 ) < size_min ) or ( t0t >= t1t ):
            p_first = 0.999
        else:
            p_first = logrank_test( t0, t1, e0, e1 ).p_value
        # last
        t0, t1, e0, e1 = get_event( data_group_last )
        size_min = int( size_min / 2 +0.999 ) 
        if ( len( t0 ) < size_min ) or ( len( t1 ) < size_min ):
            p_last = 0.999
        else: 
            p_last = -1
        # return
        return max( p_first, p_last )
    # 
    else:
        raise ValueError( ' "interval_groups" length is not supported! ' )
#
#
def calculate_individuals_fitness( population, data, num_genes_decision,
                                   is_train, size_min, n_jobs, is_warmup=False,
                                 ):
    # Get grouping information
    [ list_score, list_indices_sorted, list_interval_groups,
      list_positions_decison, list_values_decison
    ] = get_decision_information(         population = population, 
                                                data = data, 
                                  num_genes_decision = num_genes_decision, 
                                            is_train = is_train,
                                )
    ## Calculation of Differences
    if n_jobs == None:
        list_p = []
        if is_train:
            for i in range( len( list_score ) ):
                # grouping and caculate p
                p_max = calculate_difference(        
                                           data = data[ list_indices_sorted[ i ] ], 
                                interval_groups = list_interval_groups[ i ], 
                                          score = list_score[ i ],
                                       is_train = is_train,
                                       size_min = size_min, 
                                       is_warmup = is_warmup,
                                             )
                list_p.append( p_max )
        else:
            for i in range( len( list_score ) ):
                # grouping and caculate p
                p_max = calculate_difference(     data = data, 
                                       interval_groups = list_interval_groups[ i ], 
                                                 score = list_score[ i ],
                                              is_train = is_train,
                                              size_min = size_min,
                                             is_warmup = is_warmup,
                                            )
        
                list_p.append( p_max )
    else:
        if is_train:
            list_p = Parallel( n_jobs = n_jobs )(
                delayed( calculate_difference )( 
                                           data = data[ list_indices_sorted[ i ] ], 
                                interval_groups = list_interval_groups[ i ],  
                                          score = list_score[ i ],
                                       is_train = True,
                                       size_min = size_min,
                                      is_warmup = is_warmup,
                                              ) for i in range( len( list_score ) ) 
                                                )
        else:
            list_p = Parallel( n_jobs = n_jobs )(
                delayed( calculate_difference )( 
                                              data = data, 
                                   interval_groups = list_interval_groups[ i ],  
                                             score = list_score[ i ],
                                          is_train = False,
                                          size_min = size_min,
                                         is_warmup = is_warmup,  
                                              ) for i in range( len( list_score ) ) 
                                                )
    list_p = np.array( list_p )    
    #
    list_fitness = np.power( (1 - list_p), 2 )
    # list_fitness = (1 - list_p )
    return list_fitness, list_p, list_values_decison
#
#
def detector( data, population, list_values_decison, list_p, size_min, n_jobs, is_warmup=False ):     
    # No same and p < 0.05
    v = 0.05
    mask_mini     = ( list_p < v )
    ## 3 gourp
    if list_values_decison.shape[-1] > 1: 
        mask_nor_same = ( list_values_decison[:,0] != list_values_decison[:,1] )
    # 2 group
    else: 
        mask_nor_same = ( list_values_decison[:,0] == list_values_decison[:,0] )
    locs_target = np.where( ( mask_nor_same & mask_mini ) )[0]

    num_arcoss_train = len( locs_target )
    # print( num_arcoss_train )
    
    # positions to weights
    num_genes_decision = list_values_decison.shape[1]
    chromosomes_target = population[ locs_target ]
    chromosomes_target[:, -num_genes_decision: ] = \
                                    list_values_decison[ locs_target ]
    #
    [ list_fitness_valid, list_p_valid, list_values_decison 
    ] = calculate_individuals_fitness(        population = chromosomes_target, 
                                                    data = data, 
                                      num_genes_decision = num_genes_decision, 
                                                is_train = False,
                                                size_min = size_min,
                                                  n_jobs = n_jobs, 
                                                is_warmup = is_warmup,
                                       )
    
    # num_arcoss_valid = len( np.where( mask )[0] )
    # print( num_arcoss_valid )
    mask = ( list_p_valid < v )
    chromosomes = chromosomes_target[ mask ]
    p_chromosomes = list_p_valid[ mask ]
    chromosomes[:, -num_genes_decision:] = list_values_decison[ mask ]      
    #
    return chromosomes, p_chromosomes, num_arcoss_train
#
#           
class loader:
    #
    def __init__( self, num_genes ):
        self.list_chromosomes = np.zeros( [0, num_genes ] )
    #
    def load( self, chromosomes, is_unique ):
        self.list_chromosomes = np.append( self.list_chromosomes, 
                                           chromosomes, 
                                           axis=0,
                                          )
        if is_unique:
            self.unique()
    #   
    def unique( self ):
        self.list_chromosomes = np.unique( self.list_chromosomes, axis=0 )
    #
    def num( self ):
        return len( self.list_chromosomes )
    #
    def save_chromosomes( self, path_ ):
        np.save( path_, self.list_chromosomes )
#
#                      
def select_survivors( population, list_fitness_train, num_survivors ):
    sum_fitness = np.sum( list_fitness_train )
    list_hit = sum_fitness * np.random.random( num_survivors )
    cum_fitness = np.cumsum( list_fitness_train )
    locs_hit = np.argmax( (list_hit[:,None] < cum_fitness), axis=1 )
    return population[ locs_hit ], list_fitness_train[ locs_hit ]
#
#           
def get_indices_parents( population, prob_parent ):
    nrr = np.random.random
    # 
    indices_parents = np.where( nrr( len( population ) ) < prob_parent )[0]  
    indices_parents_2 = np.random.choice( indices_parents, 
                                         [2, len(indices_parents)] 
                                       )    
    return indices_parents_2
#
#
def crossover( population, indices_parents_2, prob_crossover ):
    offspring = np.copy( population )
    num_genes = offspring.shape[1]
    locs_crossover = np.random.choice( [0, 1], 
                                       size = num_genes, 
                                       p = [1-prob_crossover, prob_crossover],
                                     )
    mask = ( locs_crossover == 1 )

    temp1 = offspring[ indices_parents_2[0][:, np.newaxis], mask ]  
    temp2 = offspring[ indices_parents_2[1][:, np.newaxis], mask ]
    offspring[ indices_parents_2[1][:, np.newaxis], mask ] = temp1  
    offspring[ indices_parents_2[0][:, np.newaxis], mask ] = temp2  
    
    # (offspring == survivors).all()
    # np.where( offspring == -100 )
    
    return offspring
#
#           
def mutation( population, num_genes_weight, prob_mutation_weight_gene, 
              prob_mutation_decision_gene, num_data_train 
            ):
    # -- array_weight --
    population = np.copy( population )
    array_weight_shape = population[:,:num_genes_weight].shape
    points_mutation = ( np.random.rand( array_weight_shape[0],
                                        array_weight_shape[1] 
                                      ) < prob_mutation_weight_gene
                       ) * ( population[:,:num_genes_weight] != 0 )
    perturbations = np.random.uniform( -2, 2, size = array_weight_shape )
    population[:,:num_genes_weight] += ( points_mutation * perturbations )
    
    # -- array_decision --
    array_decision_shape = population[:,num_genes_weight:].shape
    points_mutation = ( np.random.rand( array_decision_shape[0],
                                        array_decision_shape[1] 
                                      ) < prob_mutation_decision_gene
                       ) 
    r = np.random.randint( -10, 10, size=array_decision_shape )
    population[:,num_genes_weight:] += ( r * points_mutation)
    population[:,num_genes_weight:] = np.clip( population[:,num_genes_weight:],
                                               10, num_data_train-10 
                                             ) # 
    population[:,num_genes_weight:] = np.sort( population[:,num_genes_weight:], 
                                               axis=1 
                                             ) #     
    return population
#
#            
def controller( population, range_nz_wg, num_genes_decision ):
    #
    value_min = range_nz_wg[0]
    value_max = range_nz_wg[1]
    for i in range( len( population ) ):
        genes_weight = population[ i ][:-num_genes_decision]
        non_zero_indices = np.where( genes_weight != 0 )[0]
        #
        if non_zero_indices.size > value_max:
            np.put( genes_weight, 
                    np.random.choice( non_zero_indices, 
                                      non_zero_indices.size - value_max, 
                                      replace=False
                                    ),
                    0,
                  )
        elif non_zero_indices.size < value_min:
            zero_indices = np.where( genes_weight == 0)[0]
            np.put( genes_weight, 
                    zero_indices[:value_min-non_zero_indices.size], 
                    np.random.randint( -2, 2, 
                                       size = value_min-non_zero_indices.size
                                     )
                  )
    #
    population[:,:-num_genes_decision] = \
        np.clip( population[:,:-num_genes_decision], -10, 10,  )
    #
    population = np.round( population, 3 )
    return population




















