from DEnums import Type, OpCode

ORACLE = {
	Type.INT: {
		Type.INT : {
			OpCode.PLUS : Type.INT,
			OpCode.MINUS : Type.INT,
			OpCode.TIMES : Type.INT,
			OpCode.DIVIDE : Type.FLOAT,
			OpCode.GT : Type.BOOL,
			OpCode.GTE : Type.BOOL,
			OpCode.LT : Type.BOOL,
			OpCode.LTE : Type.BOOL,
			OpCode.EQUALS : Type.BOOL,
			OpCode.NEQUALS : Type.BOOL,
			OpCode.ASSIGN : Type.INT,
			OpCode.CONVOLUTIONS : Type.INT
		},
		Type.FLOAT: {
			OpCode.PLUS : Type.FLOAT,
			OpCode.MINUS : Type.FLOAT,
			OpCode.TIMES : Type.FLOAT,
			OpCode.DIVIDE : Type.FLOAT,
			OpCode.GT : Type.BOOL,
			OpCode.GTE : Type.BOOL,
			OpCode.LT : Type.BOOL,
			OpCode.LTE : Type.BOOL,
			OpCode.EQUALS : Type.BOOL,
			OpCode.NEQUALS : Type.BOOL,
			OpCode.ASSIGN : Type.INT,
			OpCode.CONVOLUTIONS : Type.INT
		},
		Type.CHAR: {
			OpCode.PLUS : Type.CHAR,
			OpCode.MINUS : Type.CHAR,
			OpCode.GT : Type.BOOL,
			OpCode.GTE : Type.BOOL,
			OpCode.LT : Type.BOOL,
			OpCode.LTE : Type.BOOL,
			OpCode.EQUALS : Type.BOOL,
			OpCode.NEQUALS : Type.BOOL,
			OpCode.ASSIGN : Type.INT
		}
	},
	Type.FLOAT: {
		Type.INT :{
			OpCode.PLUS : Type.FLOAT,
			OpCode.MINUS : Type.FLOAT,
			OpCode.TIMES : Type.FLOAT,
			OpCode.DIVIDE : Type.FLOAT,
			OpCode.GT : Type.BOOL,
			OpCode.GTE : Type.BOOL,
			OpCode.LT : Type.BOOL,
			OpCode.LTE : Type.BOOL,
			OpCode.EQUALS : Type.BOOL,
			OpCode.NEQUALS : Type.BOOL,
			OpCode.ASSIGN : Type.FLOAT,
			OpCode.CONVOLUTIONS : Type.INT
		},
		Type.FLOAT: {
			OpCode.PLUS : Type.FLOAT,
			OpCode.MINUS : Type.FLOAT,
			OpCode.TIMES : Type.FLOAT,
			OpCode.DIVIDE : Type.FLOAT,
			OpCode.GT : Type.BOOL,
			OpCode.GTE : Type.BOOL,
			OpCode.LT : Type.BOOL,
			OpCode.LTE : Type.BOOL,
			OpCode.EQUALS : Type.BOOL,
			OpCode.NEQUALS : Type.BOOL,
			OpCode.ASSIGN : Type.FLOAT,
			OpCode.CONVOLUTIONS : Type.INT
		}
	},
	Type.CHAR: {
		Type.INT :{
			OpCode.PLUS : Type.CHAR,
			OpCode.MINUS : Type.CHAR,
			OpCode.GT : Type.BOOL,
			OpCode.GTE : Type.BOOL,
			OpCode.LT : Type.BOOL,
			OpCode.LTE : Type.BOOL,
			OpCode.EQUALS : Type.BOOL,
			OpCode.NEQUALS : Type.BOOL,
			OpCode.ASSIGN : Type.CHAR
		},
		Type.CHAR: {
			OpCode.PLUS : Type.CHAR,
			OpCode.MINUS : Type.CHAR,
			OpCode.GT : Type.BOOL,
			OpCode.GTE : Type.BOOL,
			OpCode.LT : Type.BOOL,
			OpCode.LTE : Type.BOOL,
			OpCode.EQUALS : Type.BOOL,
			OpCode.NEQUALS : Type.BOOL,
			OpCode.ASSIGN : Type.CHAR
		}
	},
	Type.BOOL: {
		Type.INT :{
			OpCode.ASSIGN : Type.BOOL
		},
		Type.BOOL: {
			OpCode.GTE : Type.BOOL,
			OpCode.LT : Type.BOOL,
			OpCode.LTE : Type.BOOL,
			OpCode.EQUALS : Type.BOOL,
			OpCode.NEQUALS : Type.BOOL,
			OpCode.AND : Type.BOOL,
			OpCode.OR : Type.BOOL,
			OpCode.ASSIGN : Type.BOOL
		}
	}
}