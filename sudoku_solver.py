import numpy as np
import pandas as pd
from flask import Flask, render_template, request

def initiate_cube(sudoku):

    cube = np.ones((9,9,9))

    for i in range(9):
        for j in range(9):
            if sudoku[i][j] != 0:
                for k in range(9):
                    if k+1 != sudoku[i][j]:
                        cube[i][j][k] = 0

    cube = cube.astype(int)

    return cube

def exclude_update(sudoku, cube):
    
    sudoku_inter = sudoku.copy()
    
    for i in range(9):
        for j in range(9):
            
            if sudoku[i][j] == 0:
                
                for k in range(9):
                    if sudoku[i][k] != 0:                    # check row
                        cube[i][j][sudoku[i][k]-1] = 0
                    if sudoku[k][j] != 0:                    # check column
                        cube[i][j][sudoku[k][j]-1] = 0
                for k in range(3):                           # check block
                    for l in range(3):
                            if sudoku[(i // 3)*3 + k][(j // 3)*3 + l] != 0:
                                cube[i][j][sudoku[(i // 3)*3 + k][(j // 3)*3 + l]-1] = 0

                if np.sum(cube[i][j]) == 1:
                    sudoku_inter[i][j] = np.argmax(cube[i][j]) + 1
                    
    cube = cube.astype(int)
    
    return sudoku_inter, cube

def select_update_horizontal(sudoku, cube):

    sudoku_inter = sudoku.copy()
    
    for i in range(9):

        sub_cube = np.zeros(9).reshape(1,-1)
        sum = np.zeros(9)
        locations = np.array([0])
        
        for j in range(9):
            if sudoku[i][j] == 0:
                locations = np.append(locations, j)
                sub_cube = np.append(sub_cube, cube[i][j].reshape(1,-1), axis=0)
                sum += cube[i][j]

        for k in range(9):
            if sum[k] == 1:
                index = np.argmax(sub_cube[:,k])
                for l in range(9):
                    if l != k:
                        cube[i][locations[index]][l] = 0
                sudoku_inter[i][locations[index]] = k + 1

    cube = cube.astype(int)
    
    return sudoku_inter, cube

def select_update_vertical(sudoku, cube):

    sudoku_inter = sudoku.copy()
    
    for j in range(9):

        sub_cube = np.zeros(9).reshape(1,-1)
        sum = np.zeros(9)
        locations = np.array([0])
    
        for i in range(9):
            if sudoku[i][j] == 0:
                locations = np.append(locations, i)
                sub_cube = np.append(sub_cube, cube[i][j].reshape(1,-1), axis=0)
                sum += cube[i][j]

        for k in range(9):
            if sum[k] == 1:
                index = np.argmax(sub_cube[:,k])
                for l in range(9):
                    if l != k:
                        cube[locations[index]][j][l] = 0
                sudoku_inter[locations[index]][j] = k + 1
                
    cube = cube.astype(int)
    
    return sudoku_inter, cube

def select_update_block(sudoku, cube):

    sudoku_inter = sudoku.copy()

    for i in range(3):
        for j in range(3):

            sub_cube = np.zeros(9).reshape(1,-1)
            sum = np.zeros(9)
            locations = np.array([0])

            for k in range(3):
                for l in range(3):
                    if sudoku[i*3+k][j*3+l] == 0:
                        locations = np.append(locations, 3*k + l)
                        sub_cube = np.append(sub_cube, cube[i*3+k][j*3+l].reshape(1,-1), axis=0)
                        sum += cube[i*3+k][j*3+l]

            for m in range(9):
                if sum[m] == 1:
                    index = np.argmax(sub_cube[:,m])
                    for n in range(9):
                        if n != m:
                            cube[3*i + locations[index]//3][3*j + locations[index] - 3*(locations[index]//3)][n] = 0
                            sudoku_inter[3*i + locations[index]//3][3*j + locations[index] - 3*(locations[index]//3)] = m + 1
                
    cube = cube.astype(int)
    
    return sudoku_inter, cube

def block_infer(sudoku, cube):

    sudoku_inter = sudoku.copy()

    for i in range(3):
        for j in range(3):

            sub_cube = np.zeros(9).reshape(1,-1)
            sum = np.zeros(9)
            locations = np.array([0])

            for k in range(3):
                for l in range(3):
                    if sudoku[3*i+k][3*j+l] == 0:
                        locations = np.append(locations, 3*k + l)
                        sub_cube = np.append(sub_cube, cube[3*i+k][3*j+l].reshape(1,-1), axis=0)
                        sum += cube[3*i+k][3*j+l]

            for m in range(9):
                if sum[m] == 2:
                    index_1, index_2 = np.where(sub_cube[:,m] == 1)[0]
                    i_index_1 = locations[index_1]//3 + 3*i
                    i_index_2 = locations[index_2]//3 + 3*i
                    j_index_1 = locations[index_1] - 3*(locations[index_1]//3) + 3*j
                    j_index_2 = locations[index_2] - 3*(locations[index_2]//3) + 3*j

                    if i_index_1 == i_index_2:
                        for n in range(9):
                            if n != j_index_1 and n != j_index_2:
                                cube[i_index_1][n][m] = 0

                    if j_index_1 == j_index_2:
                        for n in range(9):
                            if n != i_index_1 and n != i_index_2:
                                cube[n][j_index_1][m] = 0
                
    for i in range(9):
        for j in range(9):
            if np.sum(cube[i][j]) == 1:
                sudoku_inter[i][j] = np.argmax(cube[i][j]) + 1
                    
    cube = cube.astype(int)
    
    return sudoku_inter, cube

def error_check(sudoku):

    status = 1
    
    for i in range(9):
        
        unique, counts = np.unique(sudoku[i,:], return_counts=True)
        non_unique_values = unique[counts > 1]
        
        if np.sum(non_unique_values) > 0:
            print('Error :-(...')
            status = 0
            break

    if status == 1:
        
        for j in range(9): 
        
            unique, counts = np.unique(sudoku[:,j], return_counts=True)
            non_unique_values = unique[counts > 1]
        
            if np.sum(non_unique_values) > 0:
                print('Error :-(...')
                status = 0
                break

    if status == 1:
    
        for i in range(3):
            for j in range(3):

                unique, counts = np.unique(sudoku[3*i:3+3*i,3*j:3+3*j], return_counts=True)
                non_unique_values = unique[counts > 1]
        
                if np.sum(non_unique_values) > 0:
                    print('Error :-(...')
                    status = 0
                    break

            if status == 0:
                break

    if status == 1:
        print('No error :-)!')   

def error_check_2(sudoku):

    status = 1
    
    for i in range(9):
        
        unique, counts = np.unique(sudoku[i,:], return_counts=True)
        non_unique_values = unique[counts > 1]
        
        if np.sum(non_unique_values) > 0:
            status = 0
            break

    if status == 1:
        
        for j in range(9): 
        
            unique, counts = np.unique(sudoku[:,j], return_counts=True)
            non_unique_values = unique[counts > 1]
        
            if np.sum(non_unique_values) > 0:
                status = 0
                break

    if status == 1:
    
        for i in range(3):
            for j in range(3):

                unique, counts = np.unique(sudoku[3*i:3+3*i,3*j:3+3*j], return_counts=True)
                non_unique_values = unique[counts > 1]
        
                if np.sum(non_unique_values) > 0:
                    status = 0
                    break

            if status == 0:
                break

    return status 

def verify(sudoku_1, sudoku_2):
    
    status = 1

    for i in range(9):
        for j in range(9):
            if sudoku_1[i][j] != 0:
                if sudoku_1[i][j] != sudoku_2[i][j]:
                    print('Original sudoku corrupted')
                    status = 0

    if status == 1:
        print('Original sudoku preserved')

def rules_solver(sudoku):
    
    import time
    start_time = time.time()

    cube = initiate_cube(sudoku)
    number_empty = np.array([82, np.sum(sudoku == 0)])
    sudoku_inter = sudoku.copy()

    flag = 1
    
    while number_empty[-1] != number_empty[-2]:
    
        sudoku_inter, cube = exclude_update(sudoku_inter, cube)
        number_empty = np.append(number_empty, np.sum(sudoku_inter == 0))
    
        status = error_check_2(sudoku_inter)
        if status == 0:
            flag = 0
            break

    while number_empty[-1] != 0:

        if flag == 0:
            break
            
        if number_empty[-1] != 0:
            
            sudoku_inter, cube = select_update_horizontal(sudoku_inter, cube)
            number_empty = np.append(number_empty, np.sum(sudoku_inter == 0))

            status = error_check_2(sudoku_inter)
            if status == 0:
                break

        if number_empty[-1] != 0:
            while number_empty[-1] != number_empty[-2]:
    
                sudoku_inter, cube = exclude_update(sudoku_inter, cube)
                number_empty = np.append(number_empty, np.sum(sudoku_inter == 0))
    
                status = error_check_2(sudoku_inter)
                if status == 0:
                    break

        if number_empty[-1] != 0:
        
            sudoku_inter, cube = select_update_vertical(sudoku_inter, cube)
            number_empty = np.append(number_empty, np.sum(sudoku_inter == 0))
        
            status = error_check_2(sudoku_inter)
            if status == 0:
                break

        if number_empty[-1] != 0:
            while number_empty[-1] != number_empty[-2]:
    
                sudoku_inter, cube = exclude_update(sudoku_inter, cube)
                number_empty = np.append(number_empty, np.sum(sudoku_inter == 0))
    
                status = error_check_2(sudoku_inter)
                if status == 0:
                    break

        if number_empty[-1] != 0:
        
            sudoku_inter, cube = select_update_block(sudoku_inter, cube)
            number_empty = np.append(number_empty, np.sum(sudoku_inter == 0))
        
            status = error_check_2(sudoku_inter)
            if status == 0:
                break

        if number_empty[-1] != 0:
            while number_empty[-1] != number_empty[-2]:
    
                sudoku_inter, cube = exclude_update(sudoku_inter, cube)
                number_empty = np.append(number_empty, np.sum(sudoku_inter == 0))
    
                status = error_check_2(sudoku_inter)
                if status == 0:
                    break

        if number_empty[-1] != 0:
        
            sudoku_inter, cube = block_infer(sudoku_inter, cube)
            number_empty = np.append(number_empty, np.sum(sudoku_inter == 0))
        
            status = error_check_2(sudoku_inter)
            if status == 0:
                break

        if number_empty[-1] != 0:
            while number_empty[-1] != number_empty[-2]:
    
                sudoku_inter, cube = exclude_update(sudoku_inter, cube)
                number_empty = np.append(number_empty, np.sum(sudoku_inter == 0))
    
                status = error_check_2(sudoku_inter)
                if status == 0:
                    break
                    
        elapsed_time = time.time() - start_time
    
        if elapsed_time > 1:
            print("Loop stopped after 1 second")
            break

    iterations = len(number_empty)-2
    empty_initial = number_empty[1]
    empty = number_empty[-1]
    
    return status, sudoku_inter, cube, iterations, empty_initial, empty

def trial_error(sudoku, cube):

    sudoku_trial = sudoku.copy()
    cube_trial = cube.copy()

    sudoku_error = sudoku.copy()
    cube_error = cube.copy()
    
    status = 0
    
    for i in range(9):
        for j in range(9):
            if np.sum(cube[i][j]).astype(int) == 2:
                row = i
                column = j
                status = 1
                break
        if status == 1:
            break

    if status == 1:
        for k in range(9):
            if cube[row][column][k] == 1:
                sudoku_trial[row][column] = k + 1
                cube_trial[row][column][k+1:9] = 0
                break

    if status == 1:
        for k in range(9):
            if cube[row][column][8-k] == 1:
                sudoku_error[row][column] = (8-k) + 1
                cube_error[row][column][0:8-k] = 0
                break

    return sudoku_trial, sudoku_error, cube_trial, cube_error

def sudoku_solver_4(sudoku):

    iters = 0
    
    status, sudoku_inter, cube,  iterations, empty_initial, empty = rules_solver(sudoku)

    if empty == 0:
        return sudoku_inter
        
    else:

        sudoku_1, sudoku_2, cube_1, cube_2 = trial_error(sudoku_inter, cube)
        status, sudoku_inter, cube, iterations, empty_initial, empty = rules_solver(sudoku_1)
        
        if status == 1:
            return sudoku_inter

        else:
            status, sudoku_inter, cube, iterations, empty_initial, empty = rules_solver(sudoku_2)
            return sudoku_inter

sudoku_solver = Flask(__name__)

sudoku_start = np.zeros([9,9])

@sudoku_solver.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        print("POST request received!") 
        print("Form Data Received:", request.form)

        sudoku_start[0][0] = request.form.get('num11', type=int, default=0) or 0
        sudoku_start[0][1] = request.form.get('num12', type=int, default=0) or 0
        sudoku_start[0][2] = request.form.get('num13', type=int, default=0) or 0
        sudoku_start[0][3] = request.form.get('num14', type=int, default=0) or 0
        sudoku_start[0][4] = request.form.get('num15', type=int, default=0) or 0
        sudoku_start[0][5] = request.form.get('num16', type=int, default=0) or 0
        sudoku_start[0][6] = request.form.get('num17', type=int, default=0) or 0
        sudoku_start[0][7] = request.form.get('num18', type=int, default=0) or 0
        sudoku_start[0][8] = request.form.get('num19', type=int, default=0) or 0

        sudoku_start[1][0] = request.form.get('num21', type=int, default=0) or 0
        sudoku_start[1][1] = request.form.get('num22', type=int, default=0) or 0
        sudoku_start[1][2] = request.form.get('num23', type=int, default=0) or 0
        sudoku_start[1][3] = request.form.get('num24', type=int, default=0) or 0
        sudoku_start[1][4] = request.form.get('num25', type=int, default=0) or 0
        sudoku_start[1][5] = request.form.get('num26', type=int, default=0) or 0
        sudoku_start[1][6] = request.form.get('num27', type=int, default=0) or 0
        sudoku_start[1][7] = request.form.get('num28', type=int, default=0) or 0
        sudoku_start[1][8] = request.form.get('num29', type=int, default=0) or 0

        sudoku_start[2][0] = request.form.get('num31', type=int, default=0) or 0
        sudoku_start[2][1] = request.form.get('num32', type=int, default=0) or 0
        sudoku_start[2][2] = request.form.get('num33', type=int, default=0) or 0
        sudoku_start[2][3] = request.form.get('num34', type=int, default=0) or 0
        sudoku_start[2][4] = request.form.get('num35', type=int, default=0) or 0
        sudoku_start[2][5] = request.form.get('num36', type=int, default=0) or 0
        sudoku_start[2][6] = request.form.get('num37', type=int, default=0) or 0
        sudoku_start[2][7] = request.form.get('num38', type=int, default=0) or 0
        sudoku_start[2][8] = request.form.get('num39', type=int, default=0) or 0

        sudoku_start[3][0] = request.form.get('num41', type=int, default=0) or 0
        sudoku_start[3][1] = request.form.get('num42', type=int, default=0) or 0
        sudoku_start[3][2] = request.form.get('num43', type=int, default=0) or 0
        sudoku_start[3][3] = request.form.get('num44', type=int, default=0) or 0
        sudoku_start[3][4] = request.form.get('num45', type=int, default=0) or 0
        sudoku_start[3][5] = request.form.get('num46', type=int, default=0) or 0
        sudoku_start[3][6] = request.form.get('num47', type=int, default=0) or 0
        sudoku_start[3][7] = request.form.get('num48', type=int, default=0) or 0
        sudoku_start[3][8] = request.form.get('num49', type=int, default=0) or 0

        sudoku_start[4][0] = request.form.get('num51', type=int, default=0) or 0
        sudoku_start[4][1] = request.form.get('num52', type=int, default=0) or 0
        sudoku_start[4][2] = request.form.get('num53', type=int, default=0) or 0
        sudoku_start[4][3] = request.form.get('num54', type=int, default=0) or 0
        sudoku_start[4][4] = request.form.get('num55', type=int, default=0) or 0
        sudoku_start[4][5] = request.form.get('num56', type=int, default=0) or 0
        sudoku_start[4][6] = request.form.get('num57', type=int, default=0) or 0
        sudoku_start[4][7] = request.form.get('num58', type=int, default=0) or 0
        sudoku_start[4][8] = request.form.get('num59', type=int, default=0) or 0

        sudoku_start[5][0] = request.form.get('num61', type=int, default=0) or 0
        sudoku_start[5][1] = request.form.get('num62', type=int, default=0) or 0
        sudoku_start[5][2] = request.form.get('num63', type=int, default=0) or 0
        sudoku_start[5][3] = request.form.get('num64', type=int, default=0) or 0
        sudoku_start[5][4] = request.form.get('num65', type=int, default=0) or 0
        sudoku_start[5][5] = request.form.get('num66', type=int, default=0) or 0
        sudoku_start[5][6] = request.form.get('num67', type=int, default=0) or 0
        sudoku_start[5][7] = request.form.get('num68', type=int, default=0) or 0
        sudoku_start[5][8] = request.form.get('num69', type=int, default=0) or 0

        sudoku_start[6][0] = request.form.get('num71', type=int, default=0) or 0
        sudoku_start[6][1] = request.form.get('num72', type=int, default=0) or 0
        sudoku_start[6][2] = request.form.get('num73', type=int, default=0) or 0
        sudoku_start[6][3] = request.form.get('num74', type=int, default=0) or 0
        sudoku_start[6][4] = request.form.get('num75', type=int, default=0) or 0
        sudoku_start[6][5] = request.form.get('num76', type=int, default=0) or 0
        sudoku_start[6][6] = request.form.get('num77', type=int, default=0) or 0
        sudoku_start[6][7] = request.form.get('num78', type=int, default=0) or 0
        sudoku_start[6][8] = request.form.get('num79', type=int, default=0) or 0

        sudoku_start[7][0] = request.form.get('num81', type=int, default=0) or 0
        sudoku_start[7][1] = request.form.get('num82', type=int, default=0) or 0
        sudoku_start[7][2] = request.form.get('num83', type=int, default=0) or 0
        sudoku_start[7][3] = request.form.get('num84', type=int, default=0) or 0
        sudoku_start[7][4] = request.form.get('num85', type=int, default=0) or 0
        sudoku_start[7][5] = request.form.get('num86', type=int, default=0) or 0
        sudoku_start[7][6] = request.form.get('num87', type=int, default=0) or 0
        sudoku_start[7][7] = request.form.get('num88', type=int, default=0) or 0
        sudoku_start[7][8] = request.form.get('num89', type=int, default=0) or 0

        sudoku_start[8][0] = request.form.get('num91', type=int, default=0) or 0
        sudoku_start[8][1] = request.form.get('num92', type=int, default=0) or 0
        sudoku_start[8][2] = request.form.get('num93', type=int, default=0) or 0
        sudoku_start[8][3] = request.form.get('num94', type=int, default=0) or 0
        sudoku_start[8][4] = request.form.get('num95', type=int, default=0) or 0
        sudoku_start[8][5] = request.form.get('num96', type=int, default=0) or 0
        sudoku_start[8][6] = request.form.get('num97', type=int, default=0) or 0
        sudoku_start[8][7] = request.form.get('num98', type=int, default=0) or 0
        sudoku_start[8][8] = request.form.get('num99', type=int, default=0) or 0 
    
        print(f"num15: {sudoku_start[0][0]}, num16: {sudoku_start[0][1]}, num18: {sudoku_start[0][7]}")

        sudoku_starter = sudoku_start.astype(int)
        result = sudoku_solver_4(sudoku_starter)
        result = result.tolist()

    return render_template('index.html', result=result)

if __name__ == '__main__':
    sudoku_solver.run(debug=True)
