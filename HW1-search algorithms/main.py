from  Robot import BreadthFirstSearchRobot,UniformCostSearchRobot,WAStartRobot
import MazeProblem 
import Animation  
from Heuristics import tail_manhattan_heuristic,center_manhattan_heuristic,ShorterRobotHeuristic
from Utilities import test_robot ,solve_and_display 
from Experiments import w_experiment,shorter_robot_heuristic_experiment
from MazeProblem import compute_robot_direction

# def compute_robot_direction(head, tail):
#     direction = np.sign(head - tail)
#     assert np.sum(np.abs(direction)) == 1, f"invalid robot location. head: {head}, tail: {tail}"
#     return direction

import numpy as np
import pandas as pd

# a=solve_and_display(WAStartRobot,4,blit=False,heuristic=center_manhattan_heuristic)

# print('Question 8.2')
# maze_num=99
# test_robot(WAStartRobot, [maze_num], heuristic=tail_manhattan_heuristic)
# test_robot(WAStartRobot, [maze_num], heuristic=center_manhattan_heuristic)
# test_robot(UniformCostSearchRobot, [maze_num])

# Question Input
Check_Old_versions=False
Question_Num=10
Comparison_k=True

Compare2Official=True

if Compare2Official:
    print('Question #4')
    test_robot(BreadthFirstSearchRobot,[0,1,2,3,4,5])
    print('Question #5')
    test_robot(UniformCostSearchRobot,[0,1,2,3,4,5])
    print('Question #7')
    test_robot(WAStartRobot,[0,1,2,3,4,5],heuristic=tail_manhattan_heuristic)
    print('Question 8.2')
    test_robot(WAStartRobot,[99],heuristic=tail_manhattan_heuristic)
    test_robot(WAStartRobot, [99], heuristic=center_manhattan_heuristic)
    test_robot(UniformCostSearchRobot, [99])
    print('Question #16')
    for k in [2,4,6,8]:
        test_robot(WAStartRobot,[3,4],heuristic=ShorterRobotHeuristic,k=k)



if Check_Old_versions:
    if Question_Num==7:
    # 
        test_robot(WAStartRobot, [0,1,2,3,4,5],heuristic=tail_manhattan_heuristic)
    # 
    if Question_Num==16.1:
        for k in [2,4,6,8]:
        # for k in [2]:
            test_robot(WAStartRobot,[3,4],heuristic=ShorterRobotHeuristic,k=k)
            # test_robot(WAStartRobot,[3,4],heuristic=ShorterRobotHeuristic,k=k)
            
    # if Question_Num==8.2:
    #     a=solve_and_display(UniformCostSearchRobot, 1,blit=False)  
        # b=solve_and_display(robot, maze_index, robot_params)
 
    if Question_Num==8.2:
    # a=solve_and_display(WAStartRobot, 1,blit=False,heuristic=tail_manhattan_heuristic)         
    
        solve_tail=test_robot(WAStartRobot, [999],heuristic=tail_manhattan_heuristic)
        solve_center=test_robot(WAStartRobot, [999],heuristic=center_manhattan_heuristic)   
        solve_tail=solve_tail[0]
        solve_center=solve_center[0]
        solve_tail_path=solve_tail.path
        solve_center_path=solve_center.path 
        solve_tail=[]
        solve_center=[]
        
        print('This is tail heuristic solution')
        for n_t in solve_tail_path:
            solve_tail.append(n_t.state.hash_array)
            print(n_t.state.hash_array)
        print('This is center heuristic solution')
        for n_c in solve_center_path:
            solve_center.append(n_t.state.hash_array)
            print(n_t.state.hash_array)
            
            
    if Question_Num==16.4:
        for maze_nun in list(range(2,6)):
            shorter_robot_heuristic_experiment(maze_nun)
    if Question_Num==10:
        for maze in list(range(3)):
            w_experiment(maze)  
       # w_experiment([0,1,2]) 
       
    if Question_Num==16.11:
        test_robot(WAStartRobot,[4],heuristic=center_manhattan_heuristic)
        for k in [2,4,6,8]:
        # for k in [2]:
            test_robot(WAStartRobot,[4],heuristic=ShorterRobotHeuristic,k=k)


    if Question_Num==991:
        times_vec=[]
        for i in range(10):
            x=test_robot(WAStartRobot,[3],heuristic=center_manhattan_heuristic)
            times_vec.append(x[0].solve_time)
        # y=np.array(times_vec)
        print(times_vec)






        


























































       
    # test_robot(UniformCostSearchRobot, [0,1,2,3,4,5])
    # test_robot(UniformCostSearchRobot, [99])
    
    # #a=solve_and_display(UniformCostSearchRobot, 1,blit=False)
    # UCS_maze1=test_robot(UniformCostSearchRobot, [1])
    # BFS_maze1=test_robot(BreadthFirstSearchRobot, [1])
    # # print(f" Num. of actions in maze 1 using UCS is { len(UCS_maze1[0].path)})
    # # print(f" Num. of actions in maze 1 using BFS is { len(BFSmaze1[0].path)})

    # UCS_maze3=test_robot(UniformCostSearchRobot, [3])
    # BFS_maze3=test_robot(BreadthFirstSearchRobot, [3])
    # print(f" Num. of actions in maze 1 using UCS is { len(UCS_maze1[0].path)}")
    # print(f" Num. of actions in maze 1 using BFS is { len(BFS_maze1[0].path)}")
    # print(f" Num. of actions in maze 3 using UCS is { len(UCS_maze3[0].path)}")
    # print(f" Num. of actions in maze 3 using BFS is { len(BFS_maze3[0].path)}")
    
    
    
    
# ''' 
#     Question 6
    
#     UCS_path_len=[]
#     BFS_path_len=[]
#     mazes=list(range(6))
#     for m in mazes:
#         sol_UCS=test_robot(UniformCostSearchRobot, [m])
#         sol_BFS=test_robot(BreadthFirstSearchRobot, [m])
#         UCS_path_len.append(len(sol_UCS[0].path))
#         BFS_path_len.append(len(sol_BFS[0].path))
#     print(f"This is the length of pathes by using UCS {UCS_path_len}")
#     print(f"This is the length of pathes by using BFS {BFS_path_len}")
# '''





    # test_robot(BreadthFirstSearchRobot,[1])
    # stop=1
    
    # test_robot(WAStartRobot, [99],heuristic=tail_manhattan_heuristic)
    # test_robot(WAStartRobot, [99],heuristic=center_manhattan_heuristic)   
    
    
    
    # '''
    # Question 10 
    # '''
    
    # w_experiment(0)
    # w_experiment(1)
    # w_experiment(2)

    
    
    # test_robot(WAStartRobot, [99],heuristic=tail_manhattan_heuristic)
    # test_robot(WAStartRobot, [99],heuristic=center_manhattan_heuristic)










    # test_robot(BestFirstSearchRobot, [0])
