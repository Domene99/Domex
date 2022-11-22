
class Cuadruplo:
    '''
    Cuadruplo is the class that represents a quadruple.
    It has four attributes:
        - op: the operation of type DEnums.OpCode
        - arg1: the first argument defaulted to None
        - arg2: the second argument defaulted to None
        - res: the result defaulted to None
    __repr__ is the method that returns a string representation of the quadruple.
    '''
    def __init__(self, op, arg1 = None, arg2 = None, res = None):
        self.op = op
        self.arg1 = arg1
        self.arg2 = arg2
        self.res = res
    
    def __repr__(self):
        return f'|\t{self.op}\t|\t{self.arg1}\t|\t{self.arg2}\t|\t{self.res}\t|'