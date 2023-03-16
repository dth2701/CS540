import heapq
import copy
from tokenize import ContStr
import numpy as np
import math

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = 0
    state1 = np.reshape(from_state,(3,3))
    state2 = np.reshape(to_state,(3,3))

    for i in range(3):
        for j in range(3):
            if state1[i][j] == 0:
                continue
            position_x = int(np.where(state2 == state1[i][j])[0])
            position_y = int(np.where(state2 == state1[i][j])[1])

            distance += abs(i - position_x) + abs(j - position_y)
    return distance

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    up_list = [0,1,2,3,4,5]
    down_list = [3,4,5,6,7,8]
    left_list = [0,1,3,4,6,7]
    right_list = [1,2,4,5,7,8]

    succ_states = []
    for i in range(0,9):
        up = i - 3
        if up in up_list:
            if(state[up] == 0):
                temp = copy.deepcopy(state)
                temp[i] = 0
                temp[up] = state[i]
                if temp not in succ_states and temp != state:
                    succ_states.append(temp)

    for i in range(0,9):
        down = i + 3
        if down in down_list:
            if(state[down] == 0):
                temp = copy.deepcopy(state)
                temp[i] = 0
                temp[down] = state[i]
                if temp not in succ_states and temp != state:
                    succ_states.append(temp)

    for i in range(0,9):
        left = i - 1                
        if left in left_list:
            if(state[left] == 0):
                temp = copy.deepcopy(state)
                temp[i] = 0
                temp[left] = state[i]
                if temp not in succ_states and temp != state:
                    succ_states.append(temp)

    for i in range(0,9):   
        right = i + 1         
        if right in right_list:
            if(state[right] == 0):
                temp = copy.deepcopy(state)
                temp[i] = 0
                temp[right] = state[i]
                if temp not in succ_states and temp != state:
                    succ_states.append(temp)

    return sorted(succ_states)

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    open = []
    closed = set()
    forward = []
    backward = []

    g = 0
    max_num = 0 
    parent_index = -1
    
    h = get_manhattan_distance(state)
    cost = g + h 
    heapq.heappush(open,(cost, state, (g, h, parent_index),0))

    while len(open) != 0:
        n = heapq.heappop(open)
        closed.add(tuple(n[1]))
        if n[1] == goal_state:
            break
        
        succ_list = get_succ(n[1])
        h = get_manhattan_distance(n[1])
        parent_index = n[2][2] + 1
        g = n[2][0] + 1
        
        for succ in succ_list: 
            if tuple(succ) not in closed:
                h = get_manhattan_distance(succ)
                cost = g + h 
                heapq.heappush(open,(cost, succ, (g, h, parent_index),len(backward)))
            
            backward.append(n)
            max_num = max(max_num,len(open))
    
    parent = len(backward) - 1
    while(True):
        forward.insert(0,(n[1], n[2][1]))
        n = backward[n[3]]  
        if parent == -1:
            break
        parent = n[2][2]
   
    count = 0
    for move in forward:
        print(str(move[0]) + " h=" + str(move[1]) + " moves: " + str(count))
        count += 1 

    print("Max queue length: " + str(max_num))
if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve([3, 4, 6, 0, 0, 1, 7, 2, 5])
    print()