# Justin Mak
# Assignment 3 top 20 spotify artists
# 2025/11/22
# This program analyzes a Spotify charts dataset using different dictionary
#   (map) and priority queue implementations. It first benchmarks several
#   map designs by timing how long they take to build artist→play-count maps
#   from the dataset. It then benchmarks multiple top-k algorithms for
#   extracting the artists with the highest play counts (top 20). The results
#   are written to CSV files for later graphing and used to print the final
#   top-20 list.
# Input:  CSV files in data/ (e.g., charts5000.csv or charts.csv) with Spotify chart data.
# Output: Timing results saved as CSVs in results/ and the top-20 artists printed to the console.



# data processing imports
import pandas
import ast

# experiment imports
import time
import os
import heapq


# map imports
from datastructures.map.unsorted_table_map import UnsortedTableMap
from datastructures.map.sorted_table_map import SortedTableMap
from datastructures.map.chain_hash_map import ChainHashMap
from datastructures.map.probe_hash_map import ProbeHashMap

# search tree imports
from datastructures.search_tree.avl_tree import AVLTreeMap
from datastructures.search_tree.binary_search_tree import TreeMap

# priority queue imports
from datastructures.priority_queue.unsorted_priority_queue import UnsortedPriorityQueue
from datastructures.priority_queue.sorted_priority_queue import SortedPriorityQueue
from datastructures.priority_queue.heap_priority_queue import HeapPriorityQueue
from datastructures.priority_queue.adaptable_heap_priority_queue import AdaptableHeapPriorityQueue

# CONSTANTS
DEFAULT_RANDOM_SEED = 20241123


###############
# CORE LOGIC  #
###############

def initialize_dictionary(strategy : str):
    """Initializes an empty dictionary/map object"""
    if strategy == "unsorted_table_map":
        return UnsortedTableMap()
    elif strategy == "sorted_table_map":
        return SortedTableMap()
    elif strategy == "bst":
        # TODO: binary search tree (naive)
        return TreeMap()
    elif strategy == "avl_tree":
        # TODO: AVL tree
        return AVLTreeMap()
    elif strategy == "chain_hashmap":
        # TODO: Hashmap with chaining
        return ChainHashMap()
    elif strategy == "probe_hashmap":
        # TODO: Hashmap with Probing
        return ProbeHashMap()
    elif strategy == "builtin_dict":
        # TODO: Python's built-in dictionary
        return {}

    # unexpected value
    raise ValueError(f"Unrecognized dictionary strategy '{strategy}'.")


def populate_dictionary(dataframe, dictionary) -> dict:
    """ populate_dictionary
        Populates a dictionary (aka map) object from data in a dataframe

        dataframe:  a pandas DataFrame object representing Spotify data
        dictionary: an initialized, empty dictionary/map object

        returns a dictionary with timing data
    """
    # setup timing data dictionary
    time_stats = {}
    time_stats['total_time'] = 0.0
    time_stats['lookup_time_total_ns'] = 0.0
    time_stats['lookup_n'] = 0
    time_stats['add_time_total_ns'] = 0.0
    time_stats['add_n'] = 0
    
    # start timer for overall time
    total_time_start_s = time.process_time()
    for artist_list_str in dataframe['artists']: # for each record
        artist_list = ast.literal_eval(artist_list_str) # evaluate literal from string
        for artist in artist_list: # for each artist in the artist list

            #first lookup: check if an artist is in the data structure
            lookup_start_time = time.process_time_ns()
            artist_in_dictionary = (artist in dictionary)
            lookup_end_time = time.process_time_ns()
            time_stats['lookup_time_total_ns'] += lookup_end_time-lookup_start_time
            time_stats['lookup_n'] += 1

            # if object is in the dictionary
            if artist_in_dictionary:
                # second lookup to increment the play count
                lookup_start_time = time.process_time_ns()
                dictionary[artist] += 1
                lookup_end_time = time.process_time_ns()

                # update stats
                time_stats['lookup_time_total_ns'] += lookup_end_time-lookup_start_time
                time_stats['lookup_n'] += 1
            else:
                # not in the dictionary, need to add the artist
                add_start_time = time.process_time_ns()
                dictionary[artist] = 1
                add_end_time = time.process_time_ns()

                #update stats
                time_stats['add_time_total_ns'] += add_end_time-add_start_time
                time_stats['add_n'] += 1
    total_time_end_s = time.process_time()

    # calculate averages and total time
    time_stats['total_time'] = total_time_end_s - total_time_start_s
    time_stats['lookup_time_avg_ns'] = time_stats['lookup_time_total_ns']/time_stats['lookup_n']
    time_stats['add_time_avg_ns'] = time_stats['add_time_total_ns']/time_stats['add_n']
    return time_stats

def print_results(queue, print_output: bool):
    if print_output:
        # After run_pq finishes:
        results = []
        while len(queue) > 0:
            play_count, name = queue.remove_min()
            results.append((play_count, name))

        # Print in descending order (highest plays first)
        for play_count, name in reversed(results):
            print(f"{name}: {play_count} plays")


def find_top_k_bounded_heap(strategy: dict, k: int):
    """Return a list of (artist, count) for the top-k artists
    using a bounded min-heap of size k.

    strategy: dict mapping artist -> play count
    k: number of top items to return
    """
    if k <= 0:
        return []

    # min-heap storing (count, artist)
    heap = []

    for artist, count in strategy.items():
        # fill heap up to size k
        if len(heap) < k:
            heapq.heappush(heap, (count, artist))
        else:
        # if this artist has a higher count than the smallest in heap,
        # replace the smallest
            if count > heap[0][0]:
                heapq.heapreplace(heap, (count, artist))

    # heap now has the top-k by count, but in min-heap order
    # sort descending so index 0 is the highest count
    heap.sort(reverse=True)

    # convert (count, artist) -> (artist, count) for consistency
    return [(artist, count) for (count, artist) in heap]


def run_pq(queue, strategy: dict, stats: dict, top_k: int, queue_name: str, print_output=True):
    """
    Fills a priority queue with the top_k artists based on play counts.

    Args:
        queue: The priority queue instance to use.
        artist_map (dict): Dictionary mapping artists to play counts.
        stats (dict): Dictionary to accumulate timing statistics.
        top_k (int): Number of top artists to keep.
        print_output (bool): Whether to print output for each top artist.
    """
    if print_output:
        print(f"==== {queue_name} =====")
    for name, play_count in strategy.items():
        start_ns = time.process_time_ns()

        if len(queue) < top_k:
            queue.add(play_count, name)
        else:
            lowest = queue.min()
            if play_count > lowest[0]:
                queue.remove_min()
                queue.add(play_count, name)


        end_ns = time.process_time_ns()
        stats['operation_time_total_ns'] += end_ns - start_ns

    print_results(queue, print_output)

def find_top_k(dictionary, strategy, k=20, print_output=True) -> dict:
    """ find_top_k
        Finds the top k Spotify artists from a populated dictionary/map object

        dictionary: a populated dictionary/map object
        k: the number of top artists to find (default 20)
        print_output: boolean value indicating whether output should be printed (default True)

        returns a dictionary with timing data
    """

    # setup timing data dictionary      
    time_stats = {}
    time_stats['n'] = len(dictionary)
    time_stats['strategy'] = strategy                                                                                                           
    time_stats['k'] = k
    time_stats['total_time_s'] = 0.0
    time_stats['operation_time_total_ns'] = 0.0
    
    # start timer for overall time
    start_time_s = time.process_time()
    if strategy == "bruteforce":
        found_artists = set()
        for _ in range(k):
            
            # find the kth max
            #TODO: Implement
            opp_start_time = time.process_time_ns()
            max_artist = None
            max_count = -1

            for artist, count in dictionary.items():
                if artist not in found_artists and count > max_count:
                    max_artist = artist
                    max_count = count
            if max_artist is not None:
                found_artists.add(max_artist)
                if print_output:
                    print(f"{max_artist} has {max_count} plays.")
            
            opp_end_time = time.process_time_ns()
            # do something like this to update timing information:
            time_stats['operation_time_total_ns'] += opp_end_time - opp_start_time

    elif strategy == "unsorted_priorityqueue":
        pq = UnsortedPriorityQueue()
        run_pq(pq, dictionary, time_stats, k, "unsorted_priorityqueue")

    elif strategy == "sorted_priorityqueue":
        pq = SortedPriorityQueue()
        run_pq(pq, dictionary, time_stats, k, "sorted_priorityqueue")
        
    elif strategy == "heap_priorityqueue":
        pq = HeapPriorityQueue()
        run_pq(pq, dictionary, time_stats, k, "heap_priorityqueue")
    
    elif strategy == "adaptable_heap_priorityqueue":
        pq = AdaptableHeapPriorityQueue()
        run_pq(pq, dictionary, time_stats, k, "adaptable_heap_priorityqueue")

    elif strategy == "bounded_heap_queue":
        find_top_k_bounded_heap(dictionary, k)

    else:
        raise ValueError(f"Unrecognized top k strategy '{strategy}'.")
    
    end_time_s = time.process_time()
    
    # calculate total time
    time_stats['total_time_s'] = end_time_s - start_time_s
    
    return time_stats



#########################
# EXPERIMENT FUNCTIONS  #
#########################



def save_results_to_csv(results : list[dict], prefix : str):
    """ save_results
        saves the results of an experiment to ./results/

        results: a list a dictionaries representing each row in the results
        prefix: a string for the file prefix
    """
    if not os.path.exists("./results"):
        os.mkdir("./results")
    topk_results_df = pandas.DataFrame(results)
    topk_results_df.to_csv(f"results/{prefix.strip()}-{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv", index=False)


def dictionary_experiment_1(dataframe, print_output=True):
    """Example function that compares dictionary/map data structures
        dataframe: a pandas dataframe containing the full data set
        print_output: boolean value indicating whether output of current status should be printed (default True)
    """

    # setup experiment variables
    results = []
    experiment_step = len(dataframe)//20

    for n in range(experiment_step, len(dataframe), experiment_step):
        # setup data
        dataframe_sampled = dataframe.sample(n=n,random_state=DEFAULT_RANDOM_SEED) # sample the dataframe to be size n
        for strategy in [
            "unsorted_table_map",
            "sorted_table_map",
            "bst",
            "avl_tree",
            "chain_hashmap",
            "probe_hashmap",
            "builtin_dict",
            ]:
            dictionary = initialize_dictionary(strategy)
            # populate dictionary for given size and strategy
            result_populating = populate_dictionary(dataframe_sampled, dictionary)
            result_populating['n'] = n
            result_populating['strategy'] = strategy
            results.append(result_populating)
            
            if print_output:
                print(f"{strategy} (n={len(dataframe['artists'])}):")
                print(f"\tPopulating the dictionary took {result_populating['total_time']} seconds.")
                print(f"\t{result_populating['lookup_n']} lookups took an average of {result_populating['lookup_time_avg_ns']}ns")
                print(f"\t{result_populating['add_n']} adds took an average of {result_populating['add_time_avg_ns']}ns")
    save_results_to_csv(results, "dictionary")


def topk_experiment_1(dataframe, dictionary_strategy, print_output=True):
    """Example function that compares finding the top k Spotify artists
        dataframe: a pandas dataframe containing the full data set
        print_output: boolean value indicating whether output of current status should be printed (default True)
    """

    # setup experiment variables
    results_topk = [] #list to track results
    step_n = len(dataframe)//10 # step through 10 equal sizes of the data set

    # loop through dataset sizes
    for n in range(step_n, len(dataframe), step_n):
        # setup data
        dataframe_sampled = dataframe.sample(n=n,random_state=DEFAULT_RANDOM_SEED) # sample the dataframe to be size n
        dictionary = initialize_dictionary(dictionary_strategy)
        populate_dictionary(dataframe_sampled, dictionary) # build the dictionary
        
        #loop through strategies
        for strategy in ["bruteforce", "unsorted_priorityqueue", "sorted_priorityqueue", "heap_priorityqueue", "adaptable_heap_priorityqueue", "bounded_heap_queue"]:
            # loop through k values if relevant:
            for k in [20]:
                #run experiment
                if print_output:
                    print(f"Top-k strategy '{strategy}'\t(n={n} k={k})")
                time_stats = find_top_k(dictionary, strategy, k=k, print_output=False)
                results_topk.append(time_stats)
    # save output into a file
    save_results_to_csv(results_topk, f"topk-{dictionary_strategy}")


##################
# MAIN FUNCTION  #
##################
    
if __name__ == "__main__":

    filename = "data/charts5000.csv" # small file for debugging
    # filename = "data/charts.csv" # full data set
    
    with open(filename) as input_file:
        # read in data
        dataframe = pandas.read_csv(input_file)


        #
        # experiments
        #

        dictionary_experiment_1(dataframe)
        # topk_experiment_1(dataframe, "unsorted_table_map")
        # topk_experiment_1(dataframe, "sorted_table_map")
        # topk_experiment_1(dataframe, "bst")
        # topk_experiment_1(dataframe, "avl_tree")
        # topk_experiment_1(dataframe, "chain_hashmap")
        # topk_experiment_1(dataframe, "probe_hashmap")
        # topk_experiment_1(dataframe, "builtin_dict")

        #
        # find the top k 
        #

        # # generate the dictionary
        
    filename = "data/charts5000.csv" # small file for debugging
    # filename = "data/charts.csv" # full data set
    
    with open(filename) as input_file:
        # read in data
        dataframe = pandas.read_csv(input_file)

        # find the top k
        dictionary = initialize_dictionary("builtin_dict")
        topk_experiment_1(dataframe, "builtin_dict")

        
        

   
   
