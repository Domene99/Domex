from cuadruplos import Cuadruplo
from DEnums import OpCode, Type
import numpy as np

class VirtualMachine:
    '''
    Virtual Machine is the class that will execute the quadruples.
    It will have a memory, a pointer to the current quadruple, and a list of quadruples.
    It will also have a list of functions, and a list of constants.
    It is initiated with:
        - functions: a dictionary of functions
        - constants: a dictionary of constants
        - cuadruplos: a list of quadruples
    '''
    def __init__(self, functions, constants, cuadruplos):
        self.functions = functions
        self.constants = constants
        self.cuadruplos = cuadruplos
        self.memory = {
            0: {},
            1: {},
            2: {},
            3: {}
        }
        self.memoryTemp = {
            0: {},
            1: {},
            2: {},
            3: {}
        }
        # self.defMemory defines the initial address of each scope and type
        self.defMemory = {
            # Global
            0: {
                Type.INT: 1000,
                Type.FLOAT: 2000,
                Type.CHAR: 3000,
                Type.BOOL: 4000,
            },
            # Local
            1: {
                Type.INT: 5000,
                Type.FLOAT: 6000,
                Type.CHAR: 7000,
                Type.BOOL: 8000,
            },
            # Temporal
            2: {
                Type.INT: 9000,
                Type.FLOAT: 10000,
                Type.CHAR: 11000,
                Type.BOOL: 12000,
                Type.ADDRESS: 13000,
            },
            # Constant
            3: {
                Type.INT: 14000,
                Type.FLOAT: 15000,
                Type.CHAR: 16000,
                Type.BOOL: 17000,
                Type.STRING: 18000,
            },
        }
        self.nextContext = []
        self.lastContext = []
        self.cuadruplo_pointer = 0
        self.cuad_aux = 0
        self.left_dim, self.right_dim = 0, 0
        self.currFunc = ''
        self.name = next(iter(functions))
        self.initEverything()

    def initEverything(self):
        # Populates the memory with the constants
        self.__initScope(self.name, 0)
        self.memory = self.memoryTemp
        constAux = {}
        for key, value in self.constants.items():
            if Type.INT in value['type']:
                constAux[str(value['loc'])] = int(key)
            elif Type.FLOAT in value['type']:
                constAux[str(value['loc'])] = float(key)
            elif Type.CHAR in value['type']:
                constAux[str(value['loc'])] = key
            elif Type.BOOL in value['type']:
                constAux[str(value['loc'])] = bool(key)
            elif Type.STRING in value['type']:
                constAux[str(value['loc'])] = key
        self.memory[3] = constAux
        self.__clearTemp()
    
    def run(self):
        # Executes the quadruples
        while self.cuadruplo_pointer < len(self.cuadruplos):
            self.cuad_aux = self.cuadruplo_pointer
            cuadruplo = self.cuadruplos[self.cuadruplo_pointer]
            self.cuadruplo_pointer += 1
            self.execute(cuadruplo)
    
    def __convertAddressMem(self, arg1, arg2, res):
        # Converts values of type address to the actual value in memory
        result = []
        for var in [arg1, arg2, res]:
            try:
                val = int(var)
            except:
                val = 0
            if val >= 13000 and val < 14000:
                result.append(self.memory[2][str(val)])
            else:
                result.append(var)
        arg1, arg2, res = result[0], result[1], result[2]
        return arg1, arg2, res
    
    def __getScope(self, loc):
        loc = int(loc)
        if loc < 5000:
            return 0
        elif loc < 9000:
            return 1
        elif loc < 14000:
            return 2
        else:
            return 3

    def __recoverMatrix(self, loc, x, y):
        # Recovers a matrix from memory depending on its location and dimensions
        # and converts it to a numpy array
        scope = self.__getScope(loc)
        res = []
        for i in range(x):
            row = []
            for j in range(y):
                row.append(self.memory[scope][str((int(loc) + int(i) * int(y) + int(j)))])
            res.append(row)
        return np.array(res)
    
    def __saveToMemory(self, res, loc):
        # Saves a value to memory depending on its location
        scope = self.__getScope(loc)
        if isinstance(res, np.ndarray):
            if res.ndim == 1:
                for i in range(len(res)):
                    self.memory[scope][str(int(loc) + i)] = res[i]
                return None
        for i in range(len(res)):
            for j in range(len(res[i])):
                self.memory[scope][str((int(loc) + int(i) * len(res[i]) + int(j)))] = res[i][j]
    
    def __initMemory(self, scope, type, size):
        start = self.defMemory[scope][type]

        for i in range(start, start + int(size)):
            if type == Type.INT:
                self.memoryTemp[scope][str(i)] = 0
            elif type == Type.FLOAT:
                self.memoryTemp[scope][str(i)] = 0.0
            elif type == Type.CHAR:
                self.memoryTemp[scope][str(i)] = ''
            elif type == Type.BOOL:
                self.memoryTemp[scope][str(i)] = False
            else:
                self.memoryTemp[scope][str(i)] = 0

    def __initScope(self, name, scope):
        self.__initMemory(scope, Type.INT, self.functions[name][Type.INT])
        self.__initMemory(scope, Type.FLOAT, self.functions[name][Type.FLOAT])
        self.__initMemory(scope, Type.CHAR, self.functions[name][Type.CHAR])

        self.__initMemory(2, Type.INT, self.functions[name][Type.TINT])
        self.__initMemory(2, Type.FLOAT, self.functions[name][Type.TFLOAT])
        self.__initMemory(2, Type.BOOL, self.functions[name][Type.BOOL])
        self.__initMemory(2, Type.CHAR, self.functions[name][Type.TCHAR])
        self.__initMemory(2, Type.ADDRESS, self.functions[name][Type.ADDRESS])
    
    def __clearTemp(self):
        self.memoryTemp = {
            0: {},
            1: {},
            2: {},
            3: {}
        }

    def __assignParams(self, list, params):
        # Assigns the values of the parameters to the temporary memory
        countI = self.defMemory[1][Type.INT]
        countF = self.defMemory[1][Type.FLOAT]
        countB = self.defMemory[1][Type.BOOL]
        countC = self.defMemory[1][Type.CHAR]
        for i in range(len(list)):
            if list[i] == 'i':
                self.memory[1][str(countI)] = params[i]
                countI += 1
            elif list[i] == 'f':
                self.memory[1][str(countF)] = params[i]
                countF += 1
            elif list[i] == 'b':
                self.memory[1][str(countB)] = params[i]
                countB += 1
            elif list[i] == 'c':
                self.memory[1][str(countC)] = params[i]
                countC += 1

    def __returnScope(self):
        last = self.lastContext.pop()
        self.memory[1] = last['memory'][1]
        self.memory[2] = last['memory'][2]
        self.currFunc = last['name']
        return last['cuad']

    def execute(self, cuadruplo):
        # Executes a quadruple depending on its operation
        arg1, arg2, res = self.__convertAddressMem(cuadruplo.arg1, cuadruplo.arg2, cuadruplo.res)
        res = res if cuadruplo.op != OpCode.ADDR else cuadruplo.res

        curr = Cuadruplo(cuadruplo.op, arg1, arg2, res)

        # The great debugger
        # print('----------memory----------')
        # print(self.memory)
        # print('--------------------------')
        # print(self.cuadruplo_pointer, curr)
        # print(self.cuadruplos[self.cuadruplo_pointer - 1])

        match cuadruplo.op:
            case OpCode.PLUS:
                self.add(curr)
            case OpCode.MINUS:
                self.substract(curr)
            case OpCode.TIMES:
                self.multiply(curr)
            case OpCode.DIVIDE:
                self.divide(curr)
            case OpCode.ASSIGN:
                self.assign(curr)
            case OpCode.GT:
                self.greaterThan(curr)
            case OpCode.GTE:
                self.greaterThanOrEqual(curr)
            case OpCode.LT:
                self.lessThan(curr)
            case OpCode.LTE:
                self.lessThanOrEqual(curr)
            case OpCode.EQUALS:
                self.equals(curr)
            case OpCode.NEQUALS:
                self.notEquals(curr)
            case OpCode.AND:
                self.and_(curr)
            case OpCode.OR:
                self.or_(curr)
            case OpCode.NOT:
                self.not_(curr)
            case OpCode.JUMP:
                self.goto(curr)
            case OpCode.JUMPTOF:
                self.gotof(curr)
            case OpCode.JUMPTOT:
                self.gotov(curr)
            case OpCode.PRINT:
                self.print(curr)
            case OpCode.PARAM:
                self.param(curr)
            case OpCode.GOSUB:
                self.gosub(curr)
            case OpCode.RETURN:
                self.return_(curr)
            case OpCode.ENDFUNC:
                self.endfunc(curr)
            case OpCode.VER:
                self.ver(curr)
            case OpCode.INPUT:
                self.input(curr)
            case OpCode.ADDR:
                self.address(curr)
            case OpCode.ERA:
                self.era(curr)
            case OpCode.VER:
                self.ver(curr)
            case OpCode.DIM:
                self.dim(curr)
            case OpCode.CONVOLUTIONS:
                self.convolution(curr)
        
    def add(self, cuadruplo):
        left_mat = self.__recoverMatrix(cuadruplo.arg1, self.left_dim[0], self.left_dim[1])
        right_mat = self.__recoverMatrix(cuadruplo.arg2, self.right_dim[0], self.right_dim[1])
        res = left_mat + right_mat
        self.__saveToMemory(res, cuadruplo.res)
    
    def substract(self, cuadruplo):
        left_mat = self.__recoverMatrix(cuadruplo.arg1, self.left_dim[0], self.left_dim[1])
        right_mat = self.__recoverMatrix(cuadruplo.arg2, self.right_dim[0], self.right_dim[1])
        res = left_mat - right_mat
        self.__saveToMemory(res, cuadruplo.res)
    
    def multiply(self, cuadruplo):
        left_mat = self.__recoverMatrix(cuadruplo.arg1, self.left_dim[0], self.left_dim[1])
        right_mat = self.__recoverMatrix(cuadruplo.arg2, self.right_dim[0], self.right_dim[1])
        res = left_mat * right_mat if self.right_dim == [1, 1] else np.matmul(left_mat, right_mat)
        self.__saveToMemory(res, cuadruplo.res)
    
    def divide(self, cuadruplo):
        leftScope = self.__getScope(cuadruplo.arg1)
        rightScope = self.__getScope(cuadruplo.arg2)
        res = self.__getScope(cuadruplo.res)
        self.memory[res][str(cuadruplo.res)] = self.memory[leftScope][str(cuadruplo.arg1)] / self.memory[rightScope][str(cuadruplo.arg2)]
    
    def assign(self, cuadruplo):
        right_mat = self.__recoverMatrix(cuadruplo.arg2, self.right_dim[0], self.right_dim[1])
        self.__saveToMemory(right_mat, cuadruplo.arg1)
    
    def greaterThan(self, cuadruplo):
        leftScope = self.__getScope(cuadruplo.arg1)
        rightScope = self.__getScope(cuadruplo.arg2)
        res = self.__getScope(cuadruplo.res)
        self.memory[res][str(cuadruplo.res)] = self.memory[leftScope][str(cuadruplo.arg1)] > self.memory[rightScope][str(cuadruplo.arg2)]
    
    def greaterThanOrEqual(self, cuadruplo):
        leftScope = self.__getScope(cuadruplo.arg1)
        rightScope = self.__getScope(cuadruplo.arg2)
        res = self.__getScope(cuadruplo.res)
        self.memory[res][str(cuadruplo.res)] = self.memory[leftScope][str(cuadruplo.arg1)] >= self.memory[rightScope][str(cuadruplo.arg2)]
    
    def lessThan(self, cuadruplo):
        leftScope = self.__getScope(cuadruplo.arg1)
        rightScope = self.__getScope(cuadruplo.arg2)
        res = self.__getScope(cuadruplo.res)
        self.memory[res][str(cuadruplo.res)] = self.memory[leftScope][str(cuadruplo.arg1)] < self.memory[rightScope][str(cuadruplo.arg2)]
    
    def lessThanOrEqual(self, cuadruplo):
        leftScope = self.__getScope(cuadruplo.arg1)
        rightScope = self.__getScope(cuadruplo.arg2)
        res = self.__getScope(cuadruplo.res)
        self.memory[res][str(cuadruplo.res)] = self.memory[leftScope][str(cuadruplo.arg1)] <= self.memory[rightScope][str(cuadruplo.arg2)]
    
    def equals(self, cuadruplo):
        left_scope = self.__getScope(cuadruplo.arg1)
        right_scope = self.__getScope(cuadruplo.arg2)
        res = self.__getScope(cuadruplo.res)
        self.memory[res][str(cuadruplo.res)] = self.memory[left_scope][str(cuadruplo.arg1)] == self.memory[right_scope][str(cuadruplo.arg2)]

    def notEquals(self, cuadruplo):
        left_scope = self.__getScope(cuadruplo.arg1)
        right_scope = self.__getScope(cuadruplo.arg2)
        res = self.__getScope(cuadruplo.res)
        self.memory[res][str(cuadruplo.res)] = self.memory[left_scope][str(cuadruplo.arg1)] != self.memory[right_scope][str(cuadruplo.arg2)]
    
    def and_(self, cuadruplo):
        left_scope = self.__getScope(cuadruplo.arg1)
        right_scope = self.__getScope(cuadruplo.arg2)
        res = self.__getScope(cuadruplo.res)
        self.memory[res][str(cuadruplo.res)] = self.memory[left_scope][str(cuadruplo.arg1)] and self.memory[right_scope][str(cuadruplo.arg2)]
    
    def or_(self, cuadruplo):
        left_scope = self.__getScope(cuadruplo.arg1)
        right_scope = self.__getScope(cuadruplo.arg2)
        res = self.__getScope(cuadruplo.res)
        self.memory[res][str(cuadruplo.res)] = self.memory[left_scope][str(cuadruplo.arg1)] or self.memory[right_scope][str(cuadruplo.arg2)]
    
    def not_(self, cuadruplo):
        right_scope = self.__getScope(cuadruplo.arg2)
        res = self.__getScope(cuadruplo.res)
        self.memory[res][cuadruplo.res] = not self.memory[right_scope][cuadruplo.arg2]
    
    def print(self, cuadruplo):
        if cuadruplo.arg1 == 'ENDLINE':
            print()
        else:
            scope = self.__getScope(cuadruplo.arg1)
            print(self.memory[scope][str(cuadruplo.arg1)], end=' ')
        
    def input(self, cuadruplo):
        inp = input('>>>')
        val = ''
        leftScope = self.__getScope(cuadruplo.arg1)
        try:
            val = type(self.memory[leftScope][str(cuadruplo.arg1)])(inp)
        except:
            raise TypeError('Input type does not match variable type')

        self.memory[leftScope][str(cuadruplo.arg1)] = val
    
    def goto(self, cuadruplo):
        self.cuadruplo_pointer = int(cuadruplo.res)
    
    def gotof(self, cuadruplo):
        scope = self.__getScope(cuadruplo.arg1)
        if not self.memory[scope][str(cuadruplo.arg1)]:
            self.cuadruplo_pointer = int(cuadruplo.res)
    
    def gotov(self, cuadruplo):
        scope = self.__getScope(cuadruplo.arg1)
        if self.memory[scope][str(cuadruplo.arg1)]:
            self.cuadruplo_pointer = int(cuadruplo.res)
    
    def era(self, cuadruplo):
        self.__initScope(cuadruplo.arg1, 1)
        self.nextContext.append({
            'name': cuadruplo.arg1,
            'memory': {
                1: self.memoryTemp[1],
                2: self.memoryTemp[2],
            },
            'params': []
        })
        self.__clearTemp()
    
    def param(self, cuadruplo):
        scope = self.__getScope(cuadruplo.arg1)
        self.nextContext[-1]['params'].append(self.memory[scope][str(cuadruplo.arg1)])
    
    def gosub(self, cuadruplo):
        # Example of executed quadruple interacting with memory, pointer, context and params
        self.lastContext.append({
            'name': self.currFunc,
            'memory': {
                1: self.memory[1],
                2: self.memory[2],
            },
            'cuad': self.cuad_aux + 1
        })
        current = self.nextContext.pop()
        self.currFunc = current['name']
        self.memory[1] = current['memory'][1]
        self.memory[2] = current['memory'][2]
        self.__assignParams(self.functions[self.currFunc]['params'], current['params'])
        self.cuadruplo_pointer = self.functions[self.currFunc]['start']
    
    def endfunc(self, cuadruplo):
        self.cuadruplo_pointer = self.__returnScope()
    
    def return_(self, cuadruplo):
        loc = self.functions[self.currFunc]['loc']
        resScope = self.__getScope(cuadruplo.res)
        self.memory[0][str(loc)] = self.memory[resScope][str(cuadruplo.res)]
        self.cuadruplo_pointer = self.__returnScope()
    
    def address(self, cuadruplo):
        arg1Scope = self.__getScope(cuadruplo.arg1)
        resScope = self.__getScope(cuadruplo.res)
        self.memory[resScope][str(cuadruplo.res)] = str(self.memory[arg1Scope][str(cuadruplo.arg1)] + int(cuadruplo.arg2))
    
    def ver(self, cuadruplo):
        arg1Scope, arg2Scope, resScope = self.__getScope(cuadruplo.arg1), self.__getScope(cuadruplo.arg2), self.__getScope(cuadruplo.res)
        val, valInf, valorSup = self.memory[arg1Scope][str(cuadruplo.arg1)], self.memory[arg2Scope][str(cuadruplo.arg2)], self.memory[resScope][str(cuadruplo.res)]
        if val >= valorSup or val < valInf:
            raise IndexError('Index out of bounds')
        
    def dim(self, cuadruplo):
        arg1 = cuadruplo.arg1.split(',')
        self.left_dim = [int(arg1[0]), int(arg1[1])]
        arg2 = cuadruplo.arg2.split(',')
        if len(arg2) > 1:
            self.right_dim = [int(arg2[0]), int(arg2[1])]
        res = cuadruplo.res.split(',')
        if len(res) > 1:
            self.res_dim = [int(res[0]), int(res[1])]
        
    def convolution(self, cuadruplo):
        left_mat = self.__recoverMatrix(cuadruplo.arg1, self.left_dim[0], self.left_dim[1])
        right_mat = self.__recoverMatrix(cuadruplo.arg2, self.right_dim[0], self.right_dim[1])
        res = np.convolve(list(np.concatenate(left_mat).flat), list(np.concatenate(right_mat).flat))
        self.__saveToMemory(res, cuadruplo.res)