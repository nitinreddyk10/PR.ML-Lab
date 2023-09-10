import numpy as np
import pandas as pd

#functions
def matrixinput(rows, cols):
    matrix = []
    for i in range(rows):
        #row = list(map(int, input(f"Enter space-separated elements for row {i+1}: ").split()))
        row = list(map(int, input().split()))
        matrix.append(row)
    return np.array(matrix)

def addMatrices(A,B):
    if((len(matrixA)!=len(matrixB)) or len(matrixA[0])!=len(matrixB[0])):
        raise ValueError("Matrices must have the same dimensions for addition.")    
    addnmatrix=[]
    for i in range(len(matrixA)):
        row=[]
        for j in range(len(matrixA[0])):
            row.append(matrixA[i][j]+matrixB[i][j])
        addnmatrix.append(row)
    return addnmatrix

def subtMatrices(A,B):
    if((len(matrixA)!=len(matrixB)) or len(matrixA[0])!=len(matrixB[0])):
        raise ValueError("Matrices must have the same dimensions for addition.")    
    subtMatrix=[]
    for i in range(len(matrixA)):
        row=[]
        for j in range(len(matrixA[0])):
            row.append(matrixA[i][j]-matrixB[i][j])
        subtMatrix.append(row)
    return subtMatrix

def scalarMult(matrix,mulfac):
    mulmatrix=[]
    for i in range(len(matrix)):
        row=[]
        for j in range(len(matrix[0])):
            row.append(matrix[i][j]*mulfac)
        mulmatrix.append(row)
    return mulmatrix

def elementwisemult(matrixA,matrixB):
    elemulmatrix=[]
    for i in range(len(matrixA)):
        row=[]
        for j in range(len(matrixA[0])):
            row.append(matrixA[i][j]*matrixB[i][j])
        elemulmatrix.append(row)
    return elemulmatrix

def matrix_multip(matrixA, matrixB):
    if len(matrixA[0]) != len(matrixB):
        raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")

    multiply_matrices = [[0 for _ in range(len(matrixB[0]))] for _ in range(len(matrixA))]

    for i in range(len(matrixA)):
        for j in range(len(matrixB[0])):
            for k in range(len(matrixB)):
                multiply_matrices[i][j] += matrixA[i][k] * matrixB[k][j]

    return multiply_matrices

def transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    transposematrix = [[0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transposematrix[j][i] = matrix[i][j]
    return transposematrix

def trace(matrix):
    if(len(matrix)!=len(matrix[0])):
        raise ValueError("Number of columns must be equal to the number of rows.")
    trace=0
    for i in range(len(matrix)):
        trace += matrix[i][i]
    return trace

def solve_lineareqn(A,B):
    try:
        X = np.linalg.solve(A, B)
        return X
    except np.linalg.LinAlgError:
        return None
    
def find_determinant(matrix):
    try:
        # Calculate the determinant using np.linalg.det
        determinant = np.linalg.det(matrix)
        return determinant
    except np.linalg.LinAlgError:
        return None
    
def find_inverse(matrix):
    try:
        # Calculate the inverse using np.linalg.inv
        inverse_matrix = np.linalg.inv(matrix)
        return inverse_matrix
    except np.linalg.LinAlgError:
        return None
    
def find_eigen(matrix):
    try:
        # Calculate the eigenvalues and eigenvectors using np.linalg.eig
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return eigenvalues, eigenvectors
    except np.linalg.LinAlgError:
        return None, None



def switchstatement(case):
    if case == 1:
        print(addMatrices(matrixA,matrixB))  
    elif case == 2:
        print(subtMatrices(matrixA,matrixB))
    elif case == 3:
        mulfac=int(input('Enter multiplication Factor: '))
        print(scalarMult(matrixA,mulfac))
        print(scalarMult(matrixB,mulfac))
    elif case == 4:
        print(elementwisemult(matrixA,matrixB))
    elif case == 5:
        print(matrix_multip(matrixA,matrixB))
    elif case == 6:
        print("Transpose A: \n",transpose(matrixA))
        print("Transpose B: \n",transpose(matrixB))
    elif case == 7:
        print("Trace A: ",trace(matrixA))
        print("Trace B: ",trace(matrixB))
    elif case == 8:
        solution=(solve_lineareqn(matrixA, matrixB))
        if solution is not None:
            print("x =", solution[0])
            print("y =", solution[1])
            # print("z =", solution[2]) #use if 3D matrix
        else:
            print("The system of equations has no unique solution.")
    elif case == 9:
        determinantA=find_determinant(matrixA)
        if determinantA is not None:
            print("DeterminantA:", determinantA)
        else:
            print("The matrix is not square or is singular. Determinant does not exist.")
        determinantB=find_determinant(matrixB)
        if determinantA is not None:
            print("DeterminantB:", determinantB)
        else:
            print("The matrix is not square or is singular. Determinant does not exist.")
    elif case == 10:
        inverse_matrixA = find_inverse(matrixA)
        inverse_matrixB = find_inverse(matrixB)
        print("Inverse Matrix A:")
        if inverse_matrixA is not None:
            print(inverse_matrixA)
        else:
            print("The matrix is not invertible. Inverse does not exist.")
        
        print("Inverse Matrix B:")
        if inverse_matrixB is not None:
            print(inverse_matrixB)
        else:
            print("The matrix is not invertible. Inverse does not exist.")

    elif case == 11:
        eigenvalues, eigenvectors = find_eigen(matrix)

    if eigenvalues is not None and eigenvectors is not None:
        print("Eigenvalues:")
        print(eigenvalues)
        print("\nEigenvectors:")
        print(eigenvectors)
    else:
        print("The matrix is not valid for eigenvalue/eigenvector calculation.")

#Actual code
m =int(input("Enter the value of M: "))
n =int(input("Enter the value of N: "))

print("Enter a M by N matrix 'A': ")
matrixA = matrixinput(m, n)

print("Enter a M by N matrix 'B': ")
matrixB = matrixinput(m, n)

# print("Matrix A:")
# print(matrixA)
# print("Matrix B:")
# print(matrixB)

choice = int(input("\n1.Matrix Addition\n2.Matrix Subtraction\n3.Scalar Matrix Multiplication\n4.Elementwise Matrix Multiplication\n5.Matrix Multiplication\n6.Matrix Transpose\n7.Trace of a Matrix\n8.Solve System of Linear Equations\n9.Determinant\n10.Inverse\n11.Eigen Value and Eigen Vector\n\nEnter Choice: "))
switchstatement(choice)