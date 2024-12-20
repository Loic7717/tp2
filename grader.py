import argparse
import json
import sys
import graderUtil
import pprint

printer = pprint.PrettyPrinter(width=41, compact=True)

grader = graderUtil.Grader()
submission = grader.load('tp2')

def checkOpenAnswer(question, answer):
    grader.addMessage(f'Your answer to question {question} is: {answer}')
    grader.addMessage(f'It will be graded manually.')
    return False

# Question 1
grader.addBasicPart('q1', lambda : checkOpenAnswer('1', submission.q1()), 2, description='Test answer to question 1')

grader.addBasicPart('q2', lambda : checkOpenAnswer('2', submission.q2()), 2, description='Test answer to question 2')

grader.addBasicPart('q3', lambda : checkOpenAnswer('3', submission.q3()), 2,  description='Test answer to question 3')

grader.addBasicPart('q4', lambda : checkOpenAnswer('4', submission.q4()), 2,  description='Test answer to question 4')

grader.addBasicPart('q5', lambda : checkOpenAnswer('5', submission.q5()), 2,  description='Test answer to question 5')

def test_add_source():
    sim_graph = {"A":{"B":4,"C":2,"D":3},
                "B":{"A":6,"C":-5},
                "C":{"D":1},
                "D":{}}

    src="source"
    answer = {"A":{"B":4,"C":2,"D":3},
                "B":{"A":6,"C":-5},
                "C":{"D":1},
                "D":{},
                "source":{"A":0, "B":0, "C":0, "D":0}}
    grader.addMessage(f'Testing add_source on: {printer.pformat(sim_graph)}, {src}')
    grader.addMessage(f'Expecting: {printer.pformat(answer)}')

    sim_graph_src=submission.add_source(sim_graph, src)
    
    if grader.requireIsEqual(answer, sim_graph_src):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {printer.pformat(sim_graph_src)}')
    return "Test passed"

grader.addBasicPart('q6', test_add_source, 2, description='Test add_source implementation')

def test_bellman_ford():
    sim_graph = {"A":{"B":4,"C":2,"D":3},
                "B":{"A":6,"C":-5},
                "C":{"D":1},
                "D":{}}

    src="source"
    answer = {"A":0, "B":0, "C":-5, "D":-4, "source":0}
    sim_graph_src=submission.add_source(sim_graph, src)
    grader.addMessage(f'Testing bellman_ford on: {printer.pformat(sim_graph_src)}, {src}')
    grader.addMessage(f'Expecting: {answer}')
    dist=submission.bellman_ford(sim_graph_src, src)

    if grader.requireIsEqual(answer, dist):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {dist}')

    neg_cycle_graph = {"A":{"B":-5,"C":2,"D":3},
             "B":{"A":3,"C":4},
             "C":{"D":1},
             "D":{}}
    grader.addMessage(f'Testing bellman_ford on: {printer.pformat(neg_cycle_graph)}, {src}')
    grader.addMessage( 'Expecting: None')
    neg_dist = submission.bellman_ford(neg_cycle_graph, "A")
    if grader.requireIsEqual(None, neg_dist):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {neg_dist}')

grader.addBasicPart('q7', test_bellman_ford, 2, description='Test bellman_ford implementation')

def test_rewrite_weights():
    sim_graph = {"A":{"B":4,"C":2,"D":3},
                "B":{"A":6,"C":-5},
                "C":{"D":1},
                "D":{}}

    opt_distance={'A': 0, 'B': 0, 'C': -5, 'D': -4, 'source': 0}
    answer = {'A': {'B': 4, 'C': 7, 'D': 7}, 'B': {'A': 6, 'C': 0}, 'C': {'D': 0}, 'D': {}}
    grader.addMessage(f'Testing rewrite_weights on: {sim_graph}, {opt_distance}')
    grader.addMessage(f'Expecting: {printer.pformat(answer)}')
    nonneg_graph=submission.rewrite_weights(sim_graph, opt_distance)
    if grader.requireIsEqual(answer, nonneg_graph):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {nonneg_graph}')

grader.addBasicPart('q8', test_rewrite_weights, 2, description='Test rewrite_weights implementation')

grader.addBasicPart('q9', lambda : checkOpenAnswer('9', submission.q9()), 2, description='Answer to question 9')

grader.addBasicPart('q10', lambda : checkOpenAnswer('10', submission.q10()), 2, description='Answer to question 10')

def test_all_distances():
    nonneg_graph={'A': {'B': 4, 'C': 7, 'D': 7}, 'B': {'A': 6, 'C': 0}, 'C': {'D': 0}, 'D': {}}
    answer = {('A', 'A'): 0, ('A', 'B'): 4, ('A', 'C'): 4, ('A', 'D'): 4, 
                                     ('B', 'A'): 6, ('B', 'B'): 0, ('B', 'C'): 0, ('B', 'D'): 0, 
                                     ('C', 'A'): None, ('C', 'B'): None, ('C', 'C'): 0, ('C', 'D'): 0, 
                                     ('D', 'A'): None, ('D', 'B'): None, ('D', 'C'): None, ('D', 'D'): 0}
    grader.addMessage(f'Testing all_distances on: {nonneg_graph}')
    grader.addMessage(f'Expecting: {printer.pformat(answer)}')
    dist=submission.all_distances(nonneg_graph)

    if grader.requireIsEqual(answer, dist):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {dist}')
    
grader.addBasicPart('q11', test_all_distances, 2, description='Test all_distances implementation')

def test_BF_SP_all_pairs():
    sim_graph = {"A":{"B":4,"C":2,"D":3},
             "B":{"A":6,"C":-5},
             "C":{"D":1},
             "D":{}}
    
    answer = {('A', 'A'): 0, ('A', 'B'): 4, ('A', 'C'): -1, ('A', 'D'): 0, 
              ('B', 'A'): 6, ('B', 'B'): 0, ('B', 'C'): -5, ('B', 'D'): -4, 
              ('C', 'A'): None, ('C', 'B'): None, ('C', 'C'): 0, ('C', 'D'): 1, 
              ('D', 'A'): None, ('D', 'B'): None, ('D', 'C'): None, ('D', 'D'): 0}

    grader.addMessage(f'Testing BF_SP_all_pairs on: {printer.pformat(sim_graph)}')
    grader.addMessage(f'Expecting: {printer.pformat(answer)}')
    dist=submission.BF_SP_all_pairs(sim_graph)

    if grader.requireIsEqual(answer, dist):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {dist}')

grader.addBasicPart('q12', test_BF_SP_all_pairs, 2, description='Test BF_SP_all_pairs implementation')

grader.addBasicPart('q13', lambda : checkOpenAnswer('13', submission.q13()), 2, description='Answer to question 13')

def test_closest_oven():
    toy_village = {'A': {'B': -3, 'E': 20, 'F': 30},
           'B': {'A': 6, 'C': 9, 'F': 39},
           'C': {'B': 9, 'D': 8},
           'D': {'C': 8, 'F': 50},
           'E': {'A': -10, 'F': 6},
           'F': {'A': -20, 'B': -25, 'D': -15, 'E': 6} }

    grader.addMessage(f'Testing closest_oven on: {printer.pformat(toy_village)}')
    toy_distance_dict = submission.BF_SP_all_pairs(toy_village)
    for node, node_list, answer in [('A', ['B','E','D'], (-3, "B")), 
                                    ('B', ['B','E','D'], (0, "B")),
                                    ('C', ['B','E','D'], (8, "D")),
                                    ('D', ['B','E','D'], (0, "D")),
                                    ('E', ['B','E','D'], (-19, "B")),
                                    ('F', ['B','E','D'], (-25, "B"))]:
        grader.addMessage(f'Expecting: closest oven from {node} among {node_list} is {answer}')
        if grader.requireIsEqual(answer, submission.closest_oven(node, node_list, toy_distance_dict)):
            grader.addMessage('Got it!')
        else:
            grader.addMessage(f'Got: {submission.closest_oven(node, node_list, toy_distance_dict)}')

grader.addBasicPart('q14', test_closest_oven, 2, description='Test closest_oven implementation')

def test_kcentre_value():
    toy_village = {'A': {'B': -3, 'E': 20, 'F': 30},
            'B': {'A': 6, 'C': 9, 'F': 39},
            'C': {'B': 9, 'D': 8},
            'D': {'C': 8, 'F': 50},
            'E': {'A': -10, 'F': 6},
            'F': {'A': -20, 'B': -25, 'D': -15, 'E': 6} }
    grader.addMessage(f'Testing kcentre_value on: {printer.pformat(toy_village)} and [B, E, D]')
    grader.addMessage(f'Expecting: 8')
    toy_distance_dict = submission.BF_SP_all_pairs(toy_village)
    result = submission.kcentre_value(toy_village, ['B','E','D'], toy_distance_dict)
    if grader.requireIsTrue(result == 8):
        grader.addMessage('Got it!')
    else:
        grader.addMessage(f'Got: {result}')

grader.addBasicPart('q15', test_kcentre_value, 2, description='Test kcentre_value implementation')

grader.addBasicPart('q16', lambda : checkOpenAnswer('16', submission.q16()), 2, description='Answer to question 16')

def test_greedy_algorithm():
    village = submission.read_map('village.map')

    village_distance_dict = submission.BF_SP_all_pairs(village)

    grader.addMessage(f'Testing greedy_algorithm on: {printer.pformat(village)}')
    force_d, force_h = submission.brute_force(village, list(village), 5, village_distance_dict)
    grader.addMessage(f'Got: {force_d} with houses {force_h} with brute_force algorithm')
    greed_d, greed_h = submission.greedy_algorithm(village, list(village), 5, village_distance_dict) # >>> 502!!
    grader.addMessage(f'Got: {greed_d} with houses {greed_h} with greedy_algorithm algorithm')
    if grader.requireIsTrue(force_d > 0):
        grader.addMessage(f'{force_d} > 0 as expected')
        if grader.requireIsTrue(greed_d >= force_d):
            grader.addMessage(f'{greed_d} >= {force_d} as expected')
        else:
            grader.addMessage(f'{greed_d} < {force_d} not as expected')
    else:
        grader.addMessage(f'{force_d} <= 0 not as expected')
    # if grader.requireIsTrue(submission.brute_force(village, list(village), 1, village_distance_dict) == (14, {'CL5'})):
    #     grader.addMessage('brute_force with 1 house is correct')
    # else:
    #     grader.addMessage('brute_force with 1 house is incorrect')
    # if grader.requireIsTrue(submission.brute_force(village, list(village), 5, village_distance_dict) == (2, {'CL4', 'CL7', 'CL6', 'CL5', 'CL2'})):
    #     grader.addMessage('brute_force with 5 houses is correct')
    # else:
    #     grader.addMessage('brute_force with 5 houses is incorrect')

grader.addBasicPart('q17', test_greedy_algorithm, 2, description='Test greedy_algorithm implementation')

def test_random_algorithm():
    village = submission.read_map('village.map')

    village_distance_dict = submission.BF_SP_all_pairs(village)
    # 
    best_d, best_houses = submission.random_algorithm(village, list(village), 5, village_distance_dict)
    grader.addMessage(f'Testing random_algorithm on: {printer.pformat(village)}')
    grader.addMessage(f'Got: {best_d} with houses {best_houses}')
    grader.requireIsGreaterThan(0, best_d)
    

grader.addBasicPart('q18', test_random_algorithm, 2, description='Test random_algorithm implementation')

grader.addBasicPart('q19', lambda : checkOpenAnswer('19', submission.q19()), 2, description='Answer to question 19')

if __name__ == "__main__":
    grader.grade()
