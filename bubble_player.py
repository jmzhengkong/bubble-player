import numpy as np
import collections
from heapq import heappush, heappop
from copy import deepcopy

def play(map, num_moves):
    moves_every_possi = scan_map(map)
    moves=[]
    for m in moves_every_possi:
        moves+=m
    if not moves:
        return None, 0, []
    score, move, map = click_move(map, moves, num_moves)
    return map, score, move



def generate_matrix(filename):
    """Generates a numpy array by reading a file.

        A valid map file contains a list of integers:
        # 0: empty
        # 1: yellow
        # 2: red
        # 3: green
        # 4: blue
        # 5: pink

    Args:
        filename: The path to a map file.
    Returns:
        A numpy array representing the game matrix.
    """
    map = []
    with open(filename, 'r') as fp:
        for line in fp:
            map.append([int(e) for e in line.split(" ")])
    return np.array(map)


def scan_map(matrix):
    m = matrix
    lf=[] # to store the output
    p=[] # to store position of each color


    #find the position of each color and store in p[0,[()],1,[()],...]
    i=1
    while i < 6:
        p.append(i)
        q = []
        p.append(q)
        for x in range(12):
            for y in range(11):

                e = m[x][y]
                if e == i:
                    q.append((x,y))
        i=i+1

    #store the position that are connected
    for i in range(1,len(p),2):
        uniqueList = []
        l=[]
        c = []
        p1=p[i]


        for position in p1:
            a=position[0]
            b=position[1]

            if (a,b+1) in p1 and (a+1,b) in p1 and (a+1,b+1)in p1 :

                c.append(((a, b), (a, b + 1), (a + 1, b),(a+1,b+1)))

            elif (a, b + 1) in p1 and (a + 1, b) in p1 and (a,b) not in c:
                c.append(((a, b), (a, b + 1), (a + 1, b)))

            elif (a, b + 1) in p1 and (a+1,b+1) in p1 and (a,b) not in c:
                c.append(((a, b), (a, b + 1), (a + 1, b+1)))

            elif (a,b+1) in p1 and (a,b) not in c:
                c.append(((a, b), (a, b + 1)))

            elif (a+1,b) in p1 and (a,b) not in c:
                c.append(((a,b),(a+1,b)))

        for k in c:
            for t in k:
                if t not in uniqueList:
                    uniqueList.append(t)



        lt=[]
        q=0

        for u in uniqueList:

            lt.append(u)
            while len(lt) > q:
                lt, uniqueList = mapping(q,uniqueList,lt)
                q=q+1
            l.append(lt)
            lt = []
            q=0

        lf.append(l)
    return lf[0],lf[1],lf[2],lf[3],lf[4]



def mapping(q,uniqueList,lt):
    a=lt[q][0]
    b=lt[q][1]

    if (a, b + 1) in uniqueList and (a, b + 1) not in lt:
        lt.append((a, b + 1))
        uniqueList.remove((a, b + 1))

    if (a + 1, b) in uniqueList and (a + 1, b) not in lt:
        lt.append((a + 1, b))
        uniqueList.remove((a + 1, b))

    if (a - 1, b) in uniqueList and (a - 1, b) not in lt:
        lt.append((a - 1, b))
        uniqueList.remove((a - 1, b))

    if (a, b - 1) in uniqueList and (a, b - 1) not in lt:
        lt.append((a, b - 1))
        uniqueList.remove((a, b - 1))


    return lt,uniqueList


def click_move(mat, moves, num_moves):
    """Finds the best move for current matrix.

    Args:
        mat: A numpy array representing the game matrix.
        moves: A list of moves consist of a list of x,y indexs that connected
    Returns:
        best_score: The score can get from the best move.
        best_move: The move can get best potential score.
        best_mat: The game matrix after the best move.
    """
    best_total_score = 0
    best_mat = np.array([])
    best_move = np.array([])
    best_score = 0

    for move in moves:
        move_score = len(move) * (len(move) - 1)
        temp = np.array(mat)
        colum_dict = collections.defaultdict(list)
        for x, y in move:
            if y not in colum_dict:
                colum_dict[y] = [i for i in range(temp.shape[0])]
            colum_dict[y].remove(x)
        for colum, rows in colum_dict.items():
            temp[:, colum] = np.concatenate(
                [np.zeros(temp.shape[0] - len(rows)), temp[rows, colum]])

        # get the compact score for current move of current matrix
        compact_score = compactness(temp)

        current_score = move_score + compact_score
        if current_score > best_total_score:
            best_total_score = current_score
            best_mat = temp
            best_move = move
            best_score = move_score

    return best_score, best_move, best_mat


# def find_shortest_path(distance):
#     """find the shortest path of every pair of modules and show the path
#
#     Args:
#         distance: A list contains of dictionaries of key is modules, values are
#         connected modules and distance.
#         [{0:[(1,2), (2,3)], 1:[(0, 2)]}, {}]
#     Returns:
#         shortest_path: A list of dictionaries, key is the path of two modules,
#         value is the distance.
#         biggest_path: The biggest distance of all shortest path.
#     """
#     biggest_path=0
#     shortest_path=[]
#     for dict in distance:
#         total_modules=list(dict.keys())
#         temp_block={}
#         for i, module in enumerate(total_modules):
#             modules=total_modules[i:]
#             visited=[]
#             heap=[]
#             heappush(heap, (0,module,(module,)))
#             while heap:
#                 dis, curmod, path=heappop(heap)
#                 if curmod not in visited and curmod in modules:
#                     visited.append(curmod)
#                     if len(path)>1:
#                         temp_block[path]=dis
#                         biggest_path=max(biggest_path, dis)
#                     if curmod in dict:
#                         for (m, d) in dict[curmod]:
#                             heappush(heap, (dis+d, m, path+(m,)))
#         shortest_path.append(temp_block)
#     return shortest_path, biggest_path


def compactness(mat):
    compact_score=0
    for i in range(1,6):
        compact_score+=compact(mat,i)
    print('final compact score is',compact_score)
    return compact_score

def compact(mat,i):
    modules=scan_compact_module(mat,i)
    #modules=[[(1,0),(1,1),(2,0),(3,0)],[(0,3),(0,4)],[(2,2),(2,3)],[(3,4)]]
    distance, max_dist=dist(modules)
    compactness=compute_compactness(distance, max_dist,modules)
    score=compactness/len(modules)
    print(i,'score:',score)
    return score

def dist(modules):
    #distance=[{(0,0):0,(1,1):0,(2,2):0,(3,3):0,(0,1):2,(0,2):1,(0,3):2,(1,2):1,(1,3):2,(2,3):1}]
    modules_copy=deepcopy(modules)
    feasible_blocks,feasible_indexes=find_feasible_block(modules,modules_copy)

    direct_distance=[]
    #print(len((feasible_blocks[0])))
    for f in range(len(feasible_indexes)):
        dict=find_direct_distance(feasible_blocks[f],feasible_indexes[f])
        direct_distance.append(dict)
        #print('direct distance:',dict)
    #direct_distance = [{0: [(1, 2), (2, 3)]}, {4: [(5, 1)]}]
    distance,max_dist=find_shortest_path(direct_distance)
    return distance,max_dist

def find_direct_distance(feasible_block,feasible_index):
    dict={}
    temp_mat=np.full((12, 11), -1)
    for m in range(len(feasible_block)):
        for index in feasible_block[m]:
            temp_mat[index]=feasible_index[m]
    d=0
    new_blocks=deepcopy(feasible_block)#every indexes we already have
    temp_mats=[]
    #print(temp_mat)
    for i in range(len(new_blocks)):
        tm=deepcopy(temp_mat)
        temp_mats.append(tm)
    while d<=21:
        for m in range(len(new_blocks)):
            temp=temp_mats[m]
            #if m==0:
                #print(temp)
            indicator=feasible_index[m]
            new_block=[]
            for index in new_blocks[m]:
                if index[0]+1<=11 and temp[(index[0]+1,index[1])]!=indicator:
                    if temp[(index[0]+1,index[1])]>=0:# meet another cluster
                        #print('here:',(index[0]+1,index[1]),temp[(index[0]+1,index[1])])
                        temp, dict=record((index[0]+1,index[1]), temp, dict, feasible_index[m],d,indicator)
                    else:
                        new_block.append((index[0]+1,index[1]))
                        temp[(index[0]+1,index[1])]=indicator
                if index[0]-1>=0 and temp[(index[0]-1,index[1])]!=indicator:
                    if temp[(index[0]-1,index[1])]>=0:
                        #print('here:', (index[0] - 1, index[1]), temp[(index[0] - 1, index[1])])
                        temp, dict=record((index[0]-1,index[1]), temp, dict, feasible_index[m],d,indicator)
                    else:
                        new_block.append((index[0] -1, index[1]))
                        temp[(index[0]- 1, index[1])] = indicator
                if index[1]+1<=10 and temp[(index[0],index[1]+1)]!=indicator:
                    if temp[(index[0],index[1]+1)]>=0:
                        #print('here:', (index[0], index[1]+1), temp[(index[0], index[1]+1)])
                        temp, dict=record((index[0],index[1]+1), temp, dict, feasible_index[m],d,indicator)
                    else:
                        new_block.append((index[0],index[1]+1))
                        temp[(index[0],index[1]+1)]=indicator
                if index[1]-1>=0 and temp[(index[0],index[1]-1)]!=indicator:
                    if temp[(index[0],index[1]-1)]>=0:
                        #print('here:', (index[0], index[1]-1), temp[(index[0], index[1]-1)])
                        temp, dict=record((index[0],index[1]-1), temp, dict, feasible_index[m],d,indicator)
                    else:
                        new_block.append((index[0], index[1]-1))
                        temp[(index[0], index[1]-1)] = indicator
            #print(new_blocks[m])
            new_blocks[m]=new_block
            #print(new_blocks[m])

        d+=1
        #print('dict:',dict)
    return dict

def record(index,temp_mat,dict,fi,d,indicator):
    if dict.__contains__(fi):
        #if (temp_mat[index], d) not in dict[fi]:
        dict[fi].append((temp_mat[index],d))
    else:
        dict[fi] = [(temp_mat[index], d)]
    indexs = np.where(temp_mat == temp_mat[index])
    index_list = [(indexs[0][d], indexs[1][d]) for d in range(len(indexs[0]))]
    for i in index_list:
        temp_mat[i]=indicator
    return temp_mat,dict
'''
def find_shortest_path(distance):
    """find the shortest path of every pair of modules and show the path

    Args:
        distance: A list contains of dictionaries of key is modules, values are
        connected modules and distance.
        [{0:[(1,2), (2,3)], 1:[(0, 2)]}, {}]
    Returns:
        shortest_path: A list of dictionaries, key is the path of two modules,
        value is the distance.
        biggest_path: The biggest distance of all shortest path.
    """
    biggest_path=0
    shortest_path=[]
    for dict in distance:
        modules=dict.keys()
        temp_block={}
        for module in modules:
            visited=[]
            heap=[]
            heappush(heap, (0,module,(module,)))
            while heap:
                dis, curmod, path=heappop(heap)
                if curmod not in visited:
                    visited.append(curmod)
                    if len(path)>1:
                        temp_block[path]=dis
                        biggest_path=max(biggest_path, dis)
                    if curmod in dict:
                        for (m, d) in dict[curmod]:
                            heappush(heap, (dis+d, m, path+(m,)))
        shortest_path.append(temp_block)
    return shortest_path, biggest_path
'''
def find_shortest_path(distance):
    """find the shortest path of every pair of modules and show the path

    Args:
        distance: A list contains of dictionaries of key is modules, values are
        connected modules and distance.
        [{0:[(1,2), (2,3)], 1:[(0, 2)]}, {}]
    Returns:
        shortest_path: A list of dictionaries, key is the path of two modules,
        value is the distance.
        biggest_path: The biggest distance of all shortest path.
    """
    biggest_path=0
    shortest_path=[]
    for dict in distance:
        total_modules=list(dict.keys())
        temp_block={}
        for i, module in enumerate(total_modules):
            modules=total_modules[i:]
            visited=[]
            heap=[]
            heappush(heap, (0,module,(module,)))
            while heap:
                dis, curmod, path=heappop(heap)
                if curmod not in visited and curmod in modules:
                    visited.append(curmod)
                    if len(path)>1:
                        temp_block[path]=dis
                        biggest_path=max(biggest_path, dis)
                    if curmod in dict:
                        for (m, d) in dict[curmod]:
                            heappush(heap, (dis+d, m, path+(m,)))
        shortest_path.append(temp_block)
    return shortest_path, biggest_path

def min_distance(module1,module2):
    min_dist=10
    if len(module1)<=len(module2):
        for m in module1:
            d=1
            m_group=[(m[0]+1,m[1]),(m[0]-1,m[1]),(m[0],m[1]+1),(m[0],m[1]-1)]
            while d<min_dist:
                m_group=spread_step_out(m_group)
                if len(set(m_group).intersection(set(module2)))>0:
                    min_dist=d
                    break
                else:
                    d+=1

    else:
        for m in module2:
            d = 1
            m_group = [(m[0]+1,m[1]),(m[0]-1,m[1]),(m[0],m[1]+1),(m[0],m[1]-1)]
            while d < min_dist:
                m_group = spread_step_out(m_group)
                if len(set(m_group).intersection(set(module1))) > 0:
                    min_dist = d
                    break
                else:
                    d += 1
    return min_dist

def spread_step_out(m_group):
    new_m_group=[]
    for m in m_group:
        new_m_group.extend([(m[0]+1,m[1]),(m[0]-1,m[1]),(m[0],m[1]+1),(m[0],m[1]-1)])
    return new_m_group

def find_feasible_block(modules,modules_copy):
    # find the feasible blocks in the matrix, so that we only calculate meaningful distances

    feasible_blocks=[]
    feasible_indexes=[]
    old_size_of_modules=0
    feasible_block=[]
    feasible_index=[]
    while len(modules_copy)>0:#till we finish all the modules

        feasible_block.append(modules_copy[0])
        feasible_index.append(modules.index(modules_copy[0]))
        modules_copy.remove(modules_copy[0])#this line has a bug
        while len(modules_copy)!=old_size_of_modules:# which means add no dot in the last round
            old_size_of_modules=len(modules_copy)
            for n in range(1,len(modules)):
                if len(modules_copy)>0 and modules[n] in modules_copy and nearby(feasible_block,modules[n]) :
                    #if a dot is close to the ones in block
                    feasible_block.append(modules[n])
                    feasible_index.append(modules.index(modules[n]))
                    modules_copy.remove(modules[n])
        feasible_blocks.append(feasible_block)
        feasible_indexes.append(feasible_index)
        feasible_block=[]
        old_size_of_modules = 0

    return feasible_blocks,feasible_indexes

def nearby(feasible_block,module):
    #check if one module is close to a block
    near=False
    for mod in feasible_block:
        for m1 in mod:
            for m2 in module:
                if abs(m1[1]-m2[1])<=1:
                    near=True
                    break
        if near==True:
            break
    return near

def compute_compactness(distance,max_dist,modules):
    compactness=0
    for i in range(len(modules)):
        compactness+=len(modules[i])*(len(modules[i])-1)
    d=1
    while d<=max_dist:
        blocks,add=get_keys(distance,d)
        #print(d,'blocks:',blocks)
        for block in blocks:
            for b in block:
                module_size=[len(modules[b[i]]) for i in range(len(b)) ]
                #print(module_size)
                if sum(module_size)<6:
                    compactness+=sum(module_size)*(sum(module_size)-1)*(1/(d))
                else:
                    compactness+=sum(module_size)*(sum(module_size)+1)*(1/(d))
        #print(d,compactness)
        d+=1
    #print('compact:',compactness)
    return compactness

def get_keys(dictionary, value):
    keys=[]
    add=False
    for dict in dictionary:
        key=[k for k, v in dict.items() if v == value]
        if len(key)>0:
            keys.append(key)
            add=True
    return keys,add

def scan_compact_module(mat,i):
    modules=[]
    indexs= np.where(mat == i)
    index_list=[(indexs[0][d],indexs[1][d]) for d in range(len(indexs[0]))]
    module = []
    m = 0
    for ind in index_list:
        module.append(ind)
        while len(module)>m:
            module,index_list=spread_check(m, module, index_list)
            m+=1
        modules.append(module)
        module=[]
        m=0
    return modules

def spread_check(m,module,index_list):
    index=module[m]
    if (index[0] - 1, index[1]) in index_list and (index[0] - 1, index[1]) not in module:
        module.append((index[0] - 1, index[1]))
        index_list.remove((index[0] - 1, index[1]))
    if (index[0] + 1, index[1]) in index_list and (index[0] + 1, index[1]) not in module:
        module.append((index[0] + 1, index[1]))
        index_list.remove((index[0] + 1, index[1]))
    if (index[0], index[1] + 1) in index_list and (index[0], index[1] + 1) not in module:
        module.append((index[0], index[1] + 1))
        index_list.remove((index[0], index[1]+1))
    if (index[0], index[1] - 1) in index_list and (index[0], index[1] - 1) not in module:
        module.append((index[0], index[1] - 1))
        index_list.remove((index[0], index[1] - 1))

    return module,index_list
