// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token

// Defines Go tokens and basic token operations.

import "strconv"

const (
	// Special tokens
	ILLEGAL = iota;
	EOF;
	COMMENT;
	
	// Identifiers and basic type literals
	// (these tokens stand for classes of literals)
	literal_beg;
	IDENT;
	INT;
	FLOAT;
	CHAR;
	STRING;
	literal_end;

	// Operators and delimiters
	operator_beg;
	ADD;
	SUB;
	MUL;
	QUO;
	REM;

	AND;
	OR;
	XOR;
	SHL;
	SHR;

	ADD_ASSIGN;
	SUB_ASSIGN;
	MUL_ASSIGN;
	QUO_ASSIGN;
	REM_ASSIGN;

	AND_ASSIGN;
	OR_ASSIGN;
	XOR_ASSIGN;
	SHL_ASSIGN;
	SHR_ASSIGN;

	LAND;
	LOR;
	ARROW;
	INC;
	DEC;

	EQL;
	LSS;
	GTR;
	ASSIGN;
	NOT;

	NEQ;
	LEQ;
	GEQ;
	DEFINE;
	ELLIPSIS;

	LPAREN;
	LBRACK;
	LBRACE;
	COMMA;
	PERIOD;

	RPAREN;
	RBRACK;
	RBRACE;
	SEMICOLON;
	COLON;
	operator_end;

	// Keywords
	keyword_beg;
	BREAK;
	CASE;
	CHAN;
	CONST;
	CONTINUE;

	DEFAULT;
	DEFER;
	ELSE;
	FALLTHROUGH;
	FOR;

	FUNC;
	GO;
	GOTO;
	IF;
	IMPORT;

	INTERFACE;
	MAP;
	PACKAGE;
	RANGE;
	RETURN;

	SELECT;
	STRUCT;
	SWITCH;
	TYPE;
	VAR;
	keyword_end;
)


// At the moment we have no array literal syntax that lets us describe
// the index for each element - use a map for now to make sure they are
// in sync.
var tokens = map [int] string {
	ILLEGAL : "ILLEGAL",

	EOF : "EOF",
	COMMENT : "COMMENT",

	IDENT : "IDENT",
	INT : "INT",
	FLOAT : "FLOAT",
	CHAR : "CHAR",
	STRING : "STRING",

	ADD : "+",
	SUB : "-",
	MUL : "*",
	QUO : "/",
	REM : "%",

	AND : "&",
	OR : "|",
	XOR : "^",
	SHL : "<<",
	SHR : ">>",

	ADD_ASSIGN : "+=",
	SUB_ASSIGN : "-=",
	MUL_ASSIGN : "+=",
	QUO_ASSIGN : "/=",
	REM_ASSIGN : "%=",

	AND_ASSIGN : "&=",
	OR_ASSIGN : "|=",
	XOR_ASSIGN : "^=",
	SHL_ASSIGN : "<<=",
	SHR_ASSIGN : ">>=",

	LAND : "&&",
	LOR : "||",
	ARROW : "<-",
	INC : "++",
	DEC : "--",

	EQL : "==",
	LSS : "<",
	GTR : ">",
	ASSIGN : "=",
	NOT : "!",

	NEQ : "!=",
	LEQ : "<=",
	GEQ : ">=",
	DEFINE : ":=",
	ELLIPSIS : "...",

	LPAREN : "(",
	LBRACK : "[",
	LBRACE : "{",
	COMMA : ",",
	PERIOD : ".",

	RPAREN : ")",
	RBRACK : "]",
	RBRACE : "}",
	SEMICOLON : ";",
	COLON : ":",

	BREAK : "break",
	CASE : "case",
	CHAN : "chan",
	CONST : "const",
	CONTINUE : "continue",

	DEFAULT : "default",
	DEFER : "defer",
	ELSE : "else",
	FALLTHROUGH : "fallthrough",
	FOR : "for",

	FUNC : "func",
	GO : "go",
	GOTO : "goto",
	IF : "if",
	IMPORT : "import",

	INTERFACE : "interface",
	MAP : "map",
	PACKAGE : "package",
	RANGE : "range",
	RETURN : "return",

	SELECT : "select",
	STRUCT : "struct",
	SWITCH : "switch",
	TYPE : "type",
	VAR : "var",
}

func TokenString(tok int) string {
	if str, exists := tokens[tok]; exists {
		return str;
	}
	return "token(" + strconv.Itoa(tok) + ")";
}


// A set of constants for precedence-based expression parsing.
// Non-operators have lowest precedence, followed by operators
// starting with precedence 0 up to unary operators and finally
// the highest precedence used for tokens used in selectors, etc.

const (
	LowestPrec = -1;  // non-operators
	UnaryPrec = 7;
	HighestPrec = 8;
)

// Returns precedence of a token. Returns LowestPrec
// if the token is not an operator.
func Precedence(tok int) int {
	switch tok {
	case COLON:
		return 0;
	case LOR:
		return 1;
	case LAND:
		return 2;
	case ARROW:
		return 3;
	case EQL, NEQ, LSS, LEQ, GTR, GEQ:
		return 4;
	case ADD, SUB, OR, XOR:
		return 5;
	case MUL, QUO, REM, SHL, SHR, AND:
		return 6;
	}
	return LowestPrec;
}


var keywords map [string] int;

func init() {
	keywords = make(map [string] int);
	for i := keyword_beg + 1; i < keyword_end; i++ {
		keywords[tokens[i]] = i;
	}
}


// Map an identifier to its keyword token or IDENT (if not a keyword).
func Lookup(ident []byte) int {
	// TODO Maps with []byte key are illegal because []byte does not
	//      support == . Should find a more efficient solution eventually.
	if tok, is_keyword := keywords[string(ident)]; is_keyword {
		return tok;
	}
	return IDENT;
}


// Predicates

// Identifiers and basic type literals
func IsLiteral(tok int) bool {
	return literal_beg < tok && tok < literal_end;
}

// Operators and delimiters
func IsOperator(tok int) bool {
	return operator_beg < tok && tok < operator_end;
}

func IsKeyword(tok int) bool {
	return keyword_beg < tok && tok < keyword_end;
}
