import random
import numpy as np


Boards = list()
Boards = [0] * 4
for i in range(4):
    Boards[i]=np.zeros([8,8])
for x in range(4):
    for j in range(8):
        Boards[x][j, random.choice(range(0,7))] = 1  

print("INITIAL BOARDS : \n", np.asarray(Boards))
board_list = [list(), list(), list(), list()]

fitnesses = list()
fitnesses = [0] * 4

def main():
    iterations = 0
    make_list(np.asarray(Boards), board_list)
    print("Each Index Representing Row\n" , np.asarray(board_list))
    while(True):
        found_flag = False
        for f in range(4):
            fitnesses[f] = fitness_function(Boards[f])
        for b in range(4):
            if fitnesses[b] == 8:
                print("\n CHESS BOARD FOUND !: ")
                print(Boards[b])
                print("\n No. of iterations : ", iterations)
                found_flag = True
                break
        if found_flag == False:
            maxindex = maxIndex(fitnesses)
            newBoards(Boards,maxindex)
            crossMutate(Boards)
            iterations += 1
            make_list(np.asarray(Boards), board_list)
            print("New Generation of Boards\n" , board_list[2], "\n",board_list[3])
        else:
            break

def make_list(Boards, board_list):
    for i in range(4):
        board_list[i]= [0] * 8 
    for x in range(4):
        for r in range(8):
            for c in range(8):
                if Boards[x][r ,c] == 1:
                    board_list[x][r] = c
                    
def fitness_function(Board):
    fitness = 0
    for i in range(8):
        temp = collisions(Board, i)
        fitness += temp
    return fitness

def verticalCheck(Board, i):
    for j in range(8):
        if(Board[i][j] == 1):
            return j
        
def maxIndex(fitness_valuess):
    max_val_index = 0
    for i in range(1,4):
        if fitness_valuess[max_val_index] < fitness_valuess[i] :
            max_val_index = i
    return max_val_index

def newBoards(Board,max_fit_val):
    for i in range(2):
        temp = Board[i]
        if i == 1:
            max_fitness_val = maxIndex(fitnesses)
            Board[i] = Board[max_fitness_val]
            Board[max_fitness_val] = temp
            fitnesses[max_fitness_val] = 0
        elif i == 0:
            Board[i] = Board[max_fit_val]
            Board[max_fit_val] = temp
            fitnesses[max_fit_val] = 0
    
def crossMutate(Board):
    array_0 = ([0] * 8)
    array_1 = ([0] * 8)
    chess1 = Board[2]
    chess2 = Board[3]
    array_0[0:4] = chess1[0:4]
    array_0[4:8] = chess2[4:8]
    array_1[0:4] = chess2[0:4]
    array_1[4:8] = chess1[4:8]
    new_0 = ([0] * 8)
    mutate(new_0,array_0)
    newcB = np.zeros((8,8))
    for i in range(8):
        newcB[i,new_0[i]] = 1
    Boards[2] = newcB
    new_1 = ([0] * 8)
    mutate(new_1,array_1)
    newcB1 = np.zeros((8,8))
    for j in range(8):
        newcB1[j,new_1[j]] = 1
    Boards[3] = newcB1
        
def mutate(c2, c3):
    for i in range(8):
        c2[i] = verticalCheck(c3,i)
    c2[random.randint(0,7)] = random.randint(0,7)

def collisions(Board, i):
    count = 0
    c = verticalCheck(Board, i)
    for j in range(8):
        if Board[j, c] == 1:
            count += 1   
    if(count >= 2):
        return 0
    row = i + 1
    col = c + 1
    while row < 8 and col < 8:
        if Board[row, col] == 1:
            return 0
        row += 1
        col += 1  
    row = i - 1
    col = c - 1
    while row >= 0 and col >= 0:
        if Board[row, col] == 1:
            return 0
        row -= 1
        col -= 1
    row = i + 1
    col = c - 1
    while row < 8 and col >= 0:
        if Board[row, col] == 1:
            return 0
        row += 1
        col -= 1 
    row = i - 1
    col = c + 1
    while row >= 0 and col < 8:
        if Board[row, col] == 1:
            return 0
        row -= 1
        col += 1
    return 1

main()