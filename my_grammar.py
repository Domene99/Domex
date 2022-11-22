from lexyacc import lex
from lexyacc import yacc
from cuadruplos import Cuadruplo
import sys
# sys.tracebacklimit = 0

# --- Tokenizer

reserved = {
    'main': "MAIN",
    'if': "IF",
    'else': "ELSE",
    'while': "WHILE",
    'input': "INPUT",
    'print': "PRINT",
    'int': "INT",
    'float': "FLOAT",
    'bool': "BOOL",
    'char': "CHAR",
    'void': "VOID",
    'return': "RETURN",
    'function': "FUNCTION",
    'program': "PROGRAM",
    'var': "VAR",
}

# All tokens must be named in advance.
tokens = [  'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'LPAREN', 'RPAREN',
            'EQUALS', 'NEQUALS', 'LT', 'GT', 'LTE', 'GTE', 'OR', 'AND', 'ASSIGN',
            'RCURLY', 'LCURLY', 'RSQUARE', 'LSQUARE', 'SEMICOLON', 'CONSTINT',
            'CONSTFLOAT', 'CONSTCHAR', 'CONSTSTRING', 'CONSTBOOL', 'ID', 'COMMA', 'COLON', 'CONVOLUTIONS'   ]

tokens += list(reserved.values())


t_COMMA				= r'\,'
t_SEMICOLON			= r'\;'
t_COLON				= r'\:'
t_LCURLY			= r'\{'
t_RCURLY			= r'\}'
t_LSQUARE			= r'\['
t_RSQUARE			= r'\]'
t_LPAREN			= r'\('
t_RPAREN			= r'\)'
t_EQUALS			= r'\=\='
t_ASSIGN			= r'\='
t_NEQUALS			= r'\!\='
t_GT        		= r'\>'
t_GTE       		= r'\>\='
t_LT    			= r'\<'
t_LTE       		= r'\<\='
t_PLUS				= r'\+'
t_MINUS				= r'\-'
t_TIMES				= r'\*'
t_DIVIDE			= r'\/'
t_AND				= r'\&\&'
t_OR				= r'\|\|'
t_CONVOLUTIONS      = r'\.\^\.'

# Ignored characters
t_ignore = ' \t'

# CONSTANT LITERALS
def t_CONSTBOOL(t):
    r'(true|false)'
    t.value = 1 if t == "true" else 0
    return t

def t_CONSTSTRING(t):
    r'\".*\"'
    t.value = t.value[1:-1]
    return t

def t_CONSTFLOAT(t):
    r'\d+\.\d+'
    t.value = float(t.value)
    return t

def t_CONSTINT(t):
    r'\d+'
    t.value = int(t.value)
    return t

# ID
def t_ID(t):
    r'[A-Za-z_][A-Za-z_1-9]*'
    t.type = reserved.get(t.value, 'ID')
    return t

def t_CONSTCHAR(t):
    r'\'.\''
    t.value = t.value[1]
    return t

# Ignored token with an action associated with it
def t_ignore_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count('\n')

# Error handler for illegal characters
def t_error(t):
    print(f'Illegal character {t.value[0]!r}')
    t.lexer.skip(1)
    
lexer = lex.lex()

# --- Parser
from DEnums import Type, OpCode
from cube import ORACLE

funcTable = {}
varTable = {}
globalTable = {}
constTable = {}

operandStack = []
operatorStack = []
opDimStack = []
typeStack = []
jumpStack = []
loopStack = []
cuads = []
dimIdStack = []
dimStack = []
dimNumStack = []

countParams = 0
currType = Type.VOID
currFunc = ""
currParams = ""
hasReturn = False
isGlobal = True
isMatrix = False


max_per_type_per_scope = 1000
memory = {
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

memoryLimit = {
    0: {
        Type.INT: 2000,
        Type.FLOAT: 3000,
        Type.CHAR: 4000,
        Type.BOOL: 5000,
    },
    1: {
        Type.INT: 6000,
        Type.FLOAT: 7000,
        Type.CHAR: 8000,
        Type.BOOL: 9000,
    },
    2: {
        Type.INT: 10000,
        Type.FLOAT: 11000,
        Type.CHAR: 12000,
        Type.BOOL: 13000,
        Type.ADDRESS: 14000,
    },
    3: {
        Type.INT: 15000,
        Type.FLOAT: 16000,
        Type.CHAR: 17000,
        Type.BOOL: 18000,
        Type.STRING: 19000,
    },
}


def countManyMem(func, scope):
    # Assigns to the funcTable the next available memory of all types for the given scope and constant scope
    funcTable[func][Type.INT] = countMem(scope, Type.INT)
    funcTable[func][Type.FLOAT] = countMem(scope, Type.FLOAT)
    funcTable[func][Type.CHAR] = countMem(scope, Type.CHAR)
    funcTable[func][Type.BOOL] = countMem(2, Type.BOOL)
    funcTable[func][Type.TINT] = countMem(2, Type.INT)
    funcTable[func][Type.TFLOAT] = countMem(2, Type.FLOAT)
    funcTable[func][Type.TCHAR] = countMem(2, Type.CHAR)
    funcTable[func][Type.ADDRESS] = countMem(2, Type.ADDRESS)

def countMem(scope, type):
    # Returns the next available memory address for the given scope and type
    global memory, memoryLimit, max_per_type_per_scope
    return memory[scope][type] - (memoryLimit[scope][type] - max_per_type_per_scope)

def assignMem(scope, type, dim = None):
    # Assigns a memory address and size to a variable from its scope, type and dimension
    global memory, memoryLimit, max_per_type_per_scope
    loc = memory[scope][type]
    size = 1
    while dim:
        size *= dim['sup']
        dim = dim['nxt']
    if loc + size >= memoryLimit[scope][type]:
        raise MemoryError("Memory limit exceeded for type and scope")
    memory[scope][type] = size + loc
    return loc, size


def defineVar(name, dim = None):
    # Defines a variable in the current scope
    global isGlobal, currType
    if isGlobal:
        memoria, size = assignMem(0, currType, dim)
        return {'name' : name, 'loc' : memoria, 'size' : size, 'nxt' : dim}
    else:
        memoria, size = assignMem(1, currType, dim)
        return {'name' : name, 'loc' : memoria, 'size' : size, 'nxt' : dim}
    

def clearLocals():
    # Clears the memory of the local and constant scope
    global memory
    memory[1][Type.INT] = 5000
    memory[1][Type.FLOAT] = 6000
    memory[1][Type.CHAR] = 7000
    memory[1][Type.BOOL] = 8000

    memory[2][Type.INT] = 9000
    memory[2][Type.FLOAT] = 10000
    memory[2][Type.CHAR] = 11000
    memory[2][Type.BOOL] = 12000
    memory[2][Type.ADDRESS] = 13000

def cuadsOperations(lineNo):
    '''
    Performs the semantic check for the operations in the operator and operand stacks.
    Performs the dimension semantic check for the operations in the operator and operand stacks.
    Adds the corresponding cuadruplos to the cuadruplos list.
    '''
    operNorm = [OpCode.DIVIDE, OpCode.LT, OpCode.GT, OpCode.NEQUALS, OpCode.EQUALS, OpCode.LTE, OpCode.GTE, OpCode.AND, OpCode.OR]
    right_op = operandStack.pop()
    right_type = typeStack.pop()
    right_dim = opDimStack.pop()
    left_op = operandStack.pop()
    left_type = typeStack.pop()
    left_dim = opDimStack.pop()
    oper = operatorStack.pop()
    
    res_type = ORACLE[right_type][left_type][oper] if oper in ORACLE[right_type][left_type] else None

    if(not res_type):
        raise TypeError(f"Line {lineNo}: Type mismatch in operation {left_type} {oper} {right_type}")
    
    res_dim = dimensionSemantics(oper, left_dim, right_dim)
    if(res_dim == 'err') :
        raise TypeError(f"Line {lineNo}: Dimension mismatch in operation {left_dim} {oper} {right_dim}")
    if(oper not in operNorm) :
        str_left_dim = str(left_dim[0]) + ',' + str(left_dim[1])
        str_right_dim = str(right_dim[0]) + ',' + str(right_dim[1])
        str_res_dim = str(res_dim[0]) + ',' + str(res_dim[1])
        cuads.append(Cuadruplo(OpCode.DIM, str_left_dim, str_right_dim, str_res_dim))
    if(oper != OpCode.ASSIGN) :
        dimension = getDim(res_dim[0], res_dim[1])
        result, size = assignMem(2, res_type, dimension)
        typeStack.append(res_type)
        opDimStack.append(res_dim)
        operandStack.append(result)
    else:
        result = None
    cuads.append(Cuadruplo(oper, left_op, right_op, result))

def dimensionSemantics(oper, left_dim, right_dim):
    '''
    Acts as the semantic oracle for dimensioned operations.
    '''
    if oper == OpCode.PLUS or oper == OpCode.MINUS:
        if left_dim == right_dim:
            return left_dim
        elif right_dim[0] == right_dim[1] == 1:
            return left_dim
        else:
            return 'err'
    elif oper == OpCode.TIMES:
        if left_dim[1] == right_dim[0]:
            return [left_dim[0], right_dim[1]]
        elif right_dim[0] == right_dim[1] == 1:
            return left_dim
    elif oper == OpCode.ASSIGN:
        if left_dim == right_dim:
            return left_dim
        else:
            return 'err'
    elif oper == OpCode.CONVOLUTIONS:
        # print("conv", left_dim, right_dim)
        if left_dim[1] == 1 and right_dim[1] == 1:
            return [left_dim[0] + right_dim[0] - 1, 1]
        else:
            return 'err'
    elif left_dim == right_dim == [1, 1]:
        return [1, 1]
    else:
        return 'err'

def getDim(x, y):
    return {'sup': y, 'nxt': {'sup': x, 'nxt': None, 'inf': None}, 'inf': None}

def getVar(name, lineNo):
    var = ''
    if name in varTable:
        var = varTable[name]
    elif name in globalTable:
        var = globalTable[name]
    else:
        raise NameError(f"Line {lineNo}: Variable {name} not defined")
    return var

def p_program(p):
    '''program : PROGRAM define_global SEMICOLON variables define_global_vars functions main'''
    countManyMem(p[2], 0)

def p_main(p):
    '''main : MAIN resolve_jump LPAREN RPAREN LCURLY variables block RCURLY'''

def p_define_global(p):
    '''define_global : ID'''
    global currFunc, funcTable, cuads, jumpStack
    currFunc = p[1]
    funcTable[currFunc] = {'type': Type.VOID}
    cuads.append(Cuadruplo(OpCode.JUMP))
    jumpStack.append(len(cuads) - 1)
    p[0] = p[1]

def p_define_global_vars(p):
    '''define_global_vars : empty'''
    global currFunc, isGlobal, globalTable, varTable
    globalTable = varTable
    varTable = {}
    isGlobal = False

def p_resolve_jump(p):
    '''resolve_jump : empty'''
    global cuads, jumpStack
    # print("resolve_jump", jumpStack[-1], len(cuads))
    cuads[jumpStack.pop()].res = len(cuads)

def p_variables(p):
    '''variables : VAR type COLON list_ids SEMICOLON variables_1
                 | empty'''
    global varTable, funcTable
    if len(p) > 2:
        size = p[6]
        for var in p[4]:
            if var['name'] in varTable:
                raise NameError(f'Variable {var["name"]} already defined')
            if var['name'] in funcTable:
                raise NameError(f'Variable {var["name"]} already defined as function')
            varTable[var['name']] = {'type': p[2], 'nxt': var['nxt'], 'loc': var['loc']}
            size += var['size']

def p_variables_1(p):
    '''variables_1 : type COLON list_ids SEMICOLON variables_1
                   | empty'''
    global varTable, funcTable
    if len(p) > 2:
        size = p[5]
        for var in p[3]:
            # print("var", var)
            if var['name'] in varTable:
                raise NameError(f'Variable {var["name"]} already defined')
            if var['name'] in funcTable:
                raise NameError(f'Variable {var["name"]} already defined as function')
            varTable[var['name']] = {'type': p[1], 'nxt': var['nxt'], 'loc': var['loc']}
            size += var['size']
    else:
        size = p[1]
    p[0] = size

def p_list_ids(p):
    '''list_ids : identifier list'''
    array = p[2]
    object = defineVar(p[1]['name'], p[1]['nxt'])
    array.append(object)
    p[0] = array
    # print("list_ids", p[0])

def p_list(p):
    '''list : COMMA identifier list
            | empty'''
    if len(p) > 2:
        array = p[3]
        object = defineVar(p[2]['name'], p[2]['nxt'])
        array.append(object)
        p[0] = array
    else:
        p[0] = []

def p_identifier(p):
    '''identifier : ID
                  | ID LSQUARE CONSTINT RSQUARE
                  | ID LSQUARE CONSTINT RSQUARE LSQUARE CONSTINT RSQUARE'''
    dim = None
    if len(p) > 2:
        if p[3] < 0:
            raise ValueError("Array index must be positive")
        dim = {
            'sup': p[3],
            'off': 0,
            'inf': 0,
            'nxt': None
        }
    if len(p) > 5:
        if p[6] < 0 or p[3] < 0:
            raise ValueError("Array index must be positive")
        dim = {
            'sup': p[3],
            'off': p[6],
            'inf': 0,
            'nxt': {
                'sup': p[6],
                'off': 0,
                'inf': 0,
                'nxt': None
            }
        }
    p[0] = {'name': p[1], 'nxt': dim}
    # print("identifier: ", p[1], p[0])

def p_type(p):
    '''type : INT
            | FLOAT
            | CHAR
            | BOOL'''
    # print(p[1], p[0])
    global currType
    if p[1] == 'int':
        p[0] = Type.INT
    elif p[1] == 'float':
        p[0] = Type.FLOAT
    elif p[1] == 'char':
        p[0] = Type.CHAR
    elif p[1] == 'bool':
        p[0] = Type.BOOL
    currType = p[0]
    # print(p[1], p[0])


def p_functions(p):
    '''functions : FUNCTION define_function LPAREN params RPAREN LCURLY variables set_start block RCURLY clear_vars functions
                 | empty'''

def p_define_function(p):
    '''define_function : return_type ID'''
    global currFunc, funcTable, cuads, jumpStack, isGlobal, currType
    if p[2] in funcTable:
        raise NameError(f'Function {p[2]} already defined')
    funcTable[p[2]] = {
        'type': p[1],
        'params': '',
        'numParams': 0,
        'numTemp': 0,
        'start': 0,
    }
    currFunc = p[2]
    if p[1] != Type.VOID:
        isGlobal = True
        currType = p[1]
        var = defineVar(p[2])
        globalTable[p[2]] = var
        isGlobal = False
        funcTable[p[2]]['loc'] = var['loc']

def p_params(p):
    '''params : def_params variables_2
              | empty'''
    global funcTable, currFunc
    if len(p) > 2:
        funcTable[currFunc]['numParams'] = p[2] + 1

def p_def_params(p):
    '''def_params : type COLON ID'''
    global funcTable, currFunc, varTable
    # print("def_params", p[3])
    if p[3] in varTable:
        raise NameError(f'Variable {p[2]} already defined')
    loc, size = assignMem(1, p[1])
    varTable[p[3]] = {'type': p[1], 'loc': loc, 'nxt': None}
    funcTable[currFunc]['params'] = funcTable[currFunc]['params'] + p[1][0]

def p_variables_2(p):
    '''variables_2 : COMMA def_params variables_2
                   | empty'''
    global funcTable, currFunc, varTable
    # print("if", p[1])
    if len(p) > 2:
        # print("variables_2",p[3])
        p[0] = p[3] + 1
    else:
        p[0] = p[1]
        # print("else", p[0])

def p_set_start(p):
    '''set_start : empty'''
    global funcTable, currFunc, cuads
    funcTable[currFunc]['start'] = len(cuads)

def p_clear_vars(p):
    '''clear_vars : empty'''
    global varTable, funcTable, currFunc, hasReturn
    # print(json.dumps(varTable, indent=4))
    varTable = {}
    cuads.append(Cuadruplo(OpCode.ENDFUNC))
    if not hasReturn and funcTable[currFunc]['type'] != Type.VOID:
        raise ValueError(f'Function {currFunc} expects return value, but none was found')
    hasReturn = False
    countManyMem(currFunc, 1)
    clearLocals()

def p_return_type(p):
    '''return_type : INT
                   | FLOAT
                   | CHAR
                   | BOOL
                   | VOID'''
    if p[1] == 'void':
        p[0] = Type.VOID
    if p[1] == 'int':
        p[0] = Type.INT
    if p[1] == 'float':
        p[0] = Type.FLOAT
    if p[1] == 'char':
        p[0] = Type.CHAR
    if p[1] == 'bool':
        p[0] = Type.BOOL

def p_block(p):
    '''block : statements block
             | empty'''
def p_statements(p):
    '''statements : assign
                  | void_function
                  | return
                  | input
                  | output
                  | condition
                  | while'''

def p_assign(p):
    '''assign : ident_exp ASSIGN expression SEMICOLON'''
    global cuads, varTable, funcTable, currFunc, globalTable, isGlobal, currType
    # print("assign", p[1], p[2], p[3], p[4])
    operatorStack.append(p[2])
    cuadsOperations(p.lineno(1))
    p[0] = p[1]

def p_void_function(p):
    '''void_function : check_function LPAREN list_exp RPAREN SEMICOLON'''
    global cuads, varTable, funcTable, currFunc, globalTable, isGlobal, currType
    if countParams != len(currParams):
        raise ValueError(f'Function {currFunc} expects {countParams} parameters, but {len(currParams)} were found')
    cuads.append(Cuadruplo(OpCode.GOSUB, p[1]))

def p_check_function(p):
    '''check_function : ID'''
    global funcTable, currFunc, varTable, currParams, countParams
    if p[1] not in funcTable:
        raise NameError(f'Function {p[1]} is not defined')
    cuads.append(Cuadruplo(OpCode.ERA, p[1]))
    countParams = 0
    currParams = funcTable[p[1]]['params']
    p[0] = p[1]

def p_list_exp(p):
    '''list_exp : check_param list_exp_1
                | empty'''

def p_check_param(p):
    '''check_param : expression'''
    global currParams, countParams
    exp = operandStack.pop()
    exp_type = typeStack.pop()
    if currParams[countParams] == exp_type[0]:
        cuads.append(Cuadruplo(OpCode.PARAM, exp, res='param' + str(countParams)))
        countParams += 1
    else:
        raise ValueError(f'Function {currFunc} expects {currParams[countParams]} parameter, but {exp_type[0]} was found')

def p_list_exp_1(p):
    '''list_exp_1 : COMMA check_param list_exp_1
                  | empty'''

def p_return(p):
    '''return : RETURN expression SEMICOLON'''
    global cuads, varTable, funcTable, currFunc, globalTable, isGlobal, currType, hasReturn
    hasReturn = True
    op = operandStack.pop()
    type = typeStack.pop()
    func_type = funcTable[currFunc]['type']
    if func_type == type:
        cuads.append(Cuadruplo(OpCode.RETURN, res=op))
    else:
        raise TypeError(f'Function {currFunc} expects {func_type} return type, but {type} was found')

def p_input(p):
    '''input : INPUT LPAREN input_1 RPAREN SEMICOLON'''
    # print("input", p[1])

def p_input_1(p):
    '''input_1 : inp_val input_2'''

def p_input_2(p):
    '''input_2 : COMMA input_1
               | empty'''

def p_inp_val(p):
    '''inp_val : ident_exp
               | empty'''
    global cuads, varTable, funcTable, currFunc, globalTable, isGlobal, currType
    if p[1] != 0:
        typeStack.pop()
        exp = operandStack.pop()
        dim = opDimStack.pop()
        if dim != [1, 1]:
            raise ValueError(f'Cannot use array as input value')
        cuads.append(Cuadruplo(OpCode.INPUT, exp))

def p_output(p):
    '''output : PRINT LPAREN output_1 RPAREN SEMICOLON'''
    # print("output", len(p), p[3])

def p_output_1(p):
    '''output_1 : print output_2'''
    # print("output_1", len(p), p[1])

def p_output_2(p):
    '''output_2 : COMMA output_1
                | empty'''
    global cuads
    if p[1] != ',':
        cuads.append(Cuadruplo(OpCode.PRINT, 'ENDLINE'))

def p_print(p):
    '''print : expression
             | CONSTSTRING'''
    global cuads, varTable, funcTable, currFunc, globalTable, isGlobal, currType
    if p[1] == None:
        exp = operandStack.pop()
        dim = opDimStack.pop()
        if dim != [1, 1]:
            raise ValueError(f'Cannot use array as output value')
        typeStack.pop()
        cuads.append(Cuadruplo(OpCode.PRINT, exp))
    else:
        string = p[1]
        if string in constTable:
            loc = constTable[string]['loc']
        else:
            loc, size = assignMem(3, Type.STRING)
            constTable[string] = {'loc': loc, 'type': Type.STRING}
        cuads.append(Cuadruplo(OpCode.PRINT, loc))

def p_condition(p):
    '''condition : IF LPAREN expression RPAREN if_jump LCURLY block RCURLY condition_1'''
    global cuads, jumpStack
    jump_id = jumpStack.pop()
    cuads[jump_id].res = len(cuads)

def p_if_jump(p):
    '''if_jump : empty'''
    global cuads, jumpStack
    exp_type = typeStack.pop()
    if exp_type != Type.BOOL:
        raise ValueError(f'Condition must be a boolean expression')
    exp = operandStack.pop()
    cuads.append(Cuadruplo(OpCode.JUMPTOF, exp))
    jumpStack.append(len(cuads) - 1)

def p_condition_1(p):
    '''condition_1 : if_else ELSE LCURLY block RCURLY
                   | empty'''

def p_if_else(p):
    '''if_else : empty'''
    global cuads, jumpStack
    jump_id = jumpStack.pop()
    cuads.append(Cuadruplo(OpCode.JUMP))
    jumpStack.append(len(cuads) - 1)
    cuads[jump_id].res = len(cuads)

def p_while(p):
    '''while : mark_tag WHILE LPAREN expression RPAREN if_jump LCURLY block RCURLY'''
    global cuads, jumpStack
    jump_id = jumpStack.pop()
    ret = jumpStack.pop()
    cuads.append(Cuadruplo(OpCode.JUMP, res=ret))
    cuads[jump_id].res = len(cuads)

def p_mark_tag(p):
    '''mark_tag : empty'''
    global cuads, jumpStack
    jumpStack.append(len(cuads))

def p_expression(p):
    '''expression : logical
                  | logical expression_1 expression'''
    global cuads, jumpStack
    if len(p) > 2:
        # print("expression", p[1], p[2], p[3], operatorStack)
        if operatorStack[-1] == OpCode.AND or operatorStack[-1] == OpCode.OR:
            # print("popping")
            cuadsOperations(p.lineno(1))

def p_expression_1(p):
    '''expression_1 : AND
                    | OR'''
    global cuads, jumpStack
    operatorStack.append(p[1])
    # print("adding and or",operatorStack)

def p_logical(p):
    '''logical : arithmetic
               | arithmetic logical_1 logical'''
    global cuads, jumpStack
    if len(p) > 2:
        if operatorStack[-1] == OpCode.GT or operatorStack[-1] == OpCode.GTE or operatorStack[-1] == OpCode.LT or operatorStack[-1] == OpCode.LTE or operatorStack[-1] == OpCode.EQUALS or operatorStack[-1] == OpCode.NEQUALS:
            cuadsOperations(p.lineno(1))
    
def p_logical_1(p):
    '''logical_1 : GT
                 | GTE
                 | LT
                 | LTE
                 | EQUALS
                 | NEQUALS'''
    global cuads, jumpStack
    operatorStack.append(p[1])
    # print("adding gt gte lt lte equals nequals",operatorStack)

def p_arithmetic(p):
    '''arithmetic : term
                  | term arithmetic_1 arithmetic'''
    global cuads, jumpStack
    if len(p) > 2:
        if operatorStack[-1] == OpCode.PLUS or operatorStack[-1] == OpCode.MINUS:
            cuadsOperations(p.lineno(1))

def p_arithmetic_1(p):
    '''arithmetic_1 : PLUS
                    | MINUS'''
    global cuads, jumpStack
    operatorStack.append(p[1])

def p_term(p):
    '''term : specials
            | specials term_1 term'''
    global cuads, jumpStack
    if len(p) > 2:
        if operatorStack[-1] == OpCode.TIMES or operatorStack[-1] == OpCode.DIVIDE:
            # print("multiplying or dividing", operatorStack[-1])
            cuadsOperations(p.lineno(1))

def p_term_1(p):
    '''term_1 : TIMES
              | DIVIDE'''
    global cuads, jumpStack
    # print("adding times divide",operatorStack)
    operatorStack.append(p[1])
    # print("adding times divide",operatorStack)


def p_specials(p):
    '''specials : factor
                | factor specials_1 specials'''
    global cuads, jumpStack
    if len(p) > 2:
        if operatorStack[-1] == ".^.":
            # print("convolutions", operatorStack)
            cuadsOperations(p.lineno(1))

def p_specials_1(p):
    '''specials_1 : CONVOLUTIONS'''
    # print("adding convolutions",operatorStack)
    operatorStack.append(p[1])
    # print("adding convolutions",operatorStack)


def p_factor(p):
    '''factor : LPAREN expression RPAREN
              | ident_exp
              | constant
              | return_func'''

def p_ident_exp(p):
    '''ident_exp : matrix
                 | array
                 | check_id'''
    global cuads, varTable, funcTable, currFunc, globalTable, isGlobal, currType, isMatrix
    x = y = 1
    dimIdStack.pop()
    dim = dimStack.pop()
    dimNumStack.pop()
    # print("ident_exp", dim)
    if dim != None:
        x = dim['sup']
        if dim['nxt'] != None:
            y = dim['nxt']['sup']
    opDimStack.append([x, y])
    p[0] = p[1]['loc']

def p_matrix(p):
    '''matrix : check_id is_dim check_dim LSQUARE expression index_dim RSQUARE check_dim LSQUARE expression index_dim RSQUARE offset_dir'''
    p[0] = p[1]

def p_array(p):
    '''array : check_id is_dim check_dim LSQUARE expression index_dim RSQUARE offset_dir'''
    p[0] = p[1]

def p_check_id(p):
    '''check_id : ID'''
    global cuads, varTable, funcTable, currFunc, globalTable, isGlobal, currType, isMatrix
    var = getVar(p[1], p.lineno(1))
    operandStack.append(var['loc'])
    typeStack.append(var['type'])
    # print("check_id", var, p[1], p.lineno(1))
    dimIdStack.append(p[1])
    dimStack.append(var['nxt'])
    # print("dimstack", dimStack)
    dimNumStack.append(0)
    p[0] = {'loc': var['loc'], 'name': p[1]}

def p_is_dim(p):
    '''is_dim : empty'''
    global operandStack
    operandStack.pop()

def p_check_dim(p):
    '''check_dim : empty'''
    global dimIdStack, dimStack, dimNumStack
    if not dimStack[-1]:
        raise ValueError(f'Variable {dimIdStack[-1]} is of {dimNumStack[-1]} dimensions')

def p_index_dim(p):
    '''index_dim : empty'''
    global dimIdStack, dimStack, dimNumStack, currType, isMatrix
    dim = dimStack[-1]
    num_dim = dimNumStack[-1]
    type = typeStack[-1]
    # print("index_dim", dim)
    # print("num_dim, type", num_dim, type)
    if type != Type.INT:
        raise ValueError(f'Index of {dimIdStack[-1]} must be an integer at line {p.lineno(1)}')
    if dim['inf'] in constTable:
        inf = constTable[dim['inf']]['loc']
    else:
        inf, size = assignMem(3, currType)
        constTable[dim['inf']] = {'loc': inf, 'type': currType}
    if dim['sup'] in constTable:
        sup = constTable[dim['sup']]['loc']
    else:
        sup, size = assignMem(3, currType)
        constTable[dim['sup']] = {'loc': sup, 'type': currType}
    cuads.append(Cuadruplo(OpCode.VER, operandStack[-1], inf, sup))
    if dim['nxt'] != None:
        leftOp = operandStack.pop()
        res, size = assignMem(2, currType)
        if dim['off'] in constTable:
            off = constTable[dim['off']]['loc']
        else:
            off, size = assignMem(3, currType)
            constTable[dim['off']] = {'loc': off, 'type': currType}
        cuads.append(Cuadruplo(OpCode.DIM, '1,1','1,1', '1,1'))
        cuads.append(Cuadruplo(OpCode.TIMES, leftOp, off, res))
        operandStack.append(res)
        isMatrix = True
    elif isMatrix:
        arg1, arg2 = operandStack.pop(), operandStack.pop()
        res, size = assignMem(2, currType)
        cuads.append(Cuadruplo(OpCode.DIM, '1,1','1,1', '1,1'))
        cuads.append(Cuadruplo(OpCode.PLUS, arg1, arg2, res))
        operandStack.append(res)
        isMatrix = False
    dimStack[-1] = dim['nxt']
    dimNumStack[-1] = num_dim + 1

def p_offset_dir(p):
    '''offset_dir : empty'''
    global dimIdStack, dimStack, dimNumStack, currType
    arg1 = operandStack.pop()
    var = getVar(dimIdStack[-1], p.lineno)
    res, size = assignMem(2, Type.ADDRESS)
    cuads.append(Cuadruplo(OpCode.ADDR, arg1, var['loc'], res))
    operandStack.append(res)

def p_constant(p):
    '''constant : CONSTINT
                | CONSTFLOAT
                | CONSTCHAR
                | CONSTBOOL'''
    global cuads, varTable, funcTable, currFunc, globalTable, isGlobal, currType
    if isinstance(p[1], int):
        currType = Type.INT
    elif isinstance(p[1], float):
        currType = Type.FLOAT
    elif isinstance(p[1], str):
        if p[1] == "true" or p[1] == "false":
            currType = Type.BOOL
        currType = Type.CHAR

    if p[1] in constTable:
        loc = constTable[p[1]]['loc']
        # print("loc", loc)
    else:
        loc, size = assignMem(3, currType)
        constTable[p[1]] = {'loc': loc, 'type': currType}
    typeStack.append(currType)
    # print("constant", typeStack[-1], p[1], loc)
    operandStack.append(loc)
    opDimStack.append([1, 1])

def p_return_func(p):
    '''return_func : check_function LPAREN list_exp RPAREN'''
    global cuads, varTable, funcTable, currFunc, globalTable, isGlobal, currType
    if countParams == len(currParams):
        cuads.append(Cuadruplo(OpCode.GOSUB, p[1]))
        type = funcTable[p[1]]['type']
        loc = funcTable[p[1]]['loc']
        locTemp, size = assignMem(2, type)
        cuads.append(Cuadruplo(OpCode.DIM, '1,1','1,1', '1,1'))
        cuads.append(Cuadruplo(OpCode.ASSIGN, locTemp, loc))
        operandStack.append(locTemp)
        typeStack.append(type)
        # print("return_func", typeStack[-1])
    else:
        raise ValueError(f'Function {p[1]} expects {countParams} parameters')

def p_empty(p):
    '''empty :'''
    p[0] = 0
    pass

def p_error(p):
    if p:
        raise SyntaxError(f'Syntax error at line {p.lineno}, unexpected value {p.value}, expected {p.type}')
    else:
        raise EOFError('Unexpected end of file')

parser = yacc.yacc()


def run():
    '''
    Runs the virtual machine for a .domex file.
    It compiles and builds a virtual machine with the given cuadruplos,
    functions, and constants.
    Writes the output to a file named 'output.txt' for easier debugging of compilation.
    '''
    filename = input("file name: ")
    if filename.split('.')[1] != 'domex':
        raise NameError('Invalid file extension, must be .domex')
    f = open(filename, "r")
    data = f.read()
    f.close()
    parser.parse(data)
    print('File compiled successfully!')
    orig_stdout = sys.stdout
    f = open('output.txt', 'w')
    sys.stdout = f
    print(json.dumps(funcTable, indent=4))
    print(json.dumps(constTable, indent=4))
    for cuad in cuads:
        print(cuad)
    sys.stdout = orig_stdout
    f.close()
    print('-------execution-------')
    vm = VirtualMachine(funcTable, constTable, cuads)
    vm.run()

import json
from my_virtual import VirtualMachine
import sys
try:
    run()
except EOFError:
    print(EOFError)