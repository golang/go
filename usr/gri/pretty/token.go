// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token

// Defines Go tokens and basic token operations.

import "strconv";

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


func TokenString(tok int) string {
	switch tok {
	case ILLEGAL: return "ILLEGAL";

	case EOF: return "EOF";
	case COMMENT: return "COMMENT";

	case IDENT: return "IDENT";
	case INT: return "INT";
	case FLOAT: return "FLOAT";
	case CHAR: return "CHAR";
	case STRING: return "STRING";

	case ADD: return "+";
	case SUB: return "-";
	case MUL: return "*";
	case QUO: return "/";
	case REM: return "%";

	case AND: return "&";
	case OR: return "|";
	case XOR: return "^";
	case SHL: return "<<";
	case SHR: return ">>";

	case ADD_ASSIGN: return "+=";
	case SUB_ASSIGN: return "-=";
	case MUL_ASSIGN: return "+=";
	case QUO_ASSIGN: return "/=";
	case REM_ASSIGN: return "%=";

	case AND_ASSIGN: return "&=";
	case OR_ASSIGN: return "|=";
	case XOR_ASSIGN: return "^=";
	case SHL_ASSIGN: return "<<=";
	case SHR_ASSIGN: return ">>=";

	case LAND: return "&&";
	case LOR: return "||";
	case ARROW: return "<-";
	case INC: return "++";
	case DEC: return "--";

	case EQL: return "==";
	case LSS: return "<";
	case GTR: return ">";
	case ASSIGN: return "=";
	case NOT: return "!";

	case NEQ: return "!=";
	case LEQ: return "<=";
	case GEQ: return ">=";
	case DEFINE: return ":=";
	case ELLIPSIS: return "...";

	case LPAREN: return "(";
	case LBRACK: return "[";
	case LBRACE: return "{";
	case COMMA: return ",";
	case PERIOD: return ".";

	case RPAREN: return ")";
	case RBRACK: return "]";
	case RBRACE: return "}";
	case SEMICOLON: return ";";
	case COLON: return ":";

	case BREAK: return "break";
	case CASE: return "case";
	case CHAN: return "chan";
	case CONST: return "const";
	case CONTINUE: return "continue";

	case DEFAULT: return "default";
	case DEFER: return "defer";
	case ELSE: return "else";
	case FALLTHROUGH: return "fallthrough";
	case FOR: return "for";

	case FUNC: return "func";
	case GO: return "go";
	case GOTO: return "goto";
	case IF: return "if";
	case IMPORT: return "import";

	case INTERFACE: return "interface";
	case MAP: return "map";
	case PACKAGE: return "package";
	case RANGE: return "range";
	case RETURN: return "return";

	case SELECT: return "select";
	case STRUCT: return "struct";
	case SWITCH: return "switch";
	case TYPE: return "type";
	case VAR: return "var";
	}

	return "token(" + strconv.Itoa(tok) + ")";
}


const (
	LowestPrec = -1;
	UnaryPrec = 7;
	HighestPrec = 8;
)


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
		keywords[TokenString(i)] = i;
	}
}


// Map an identifier to its keyword token or IDENT (if not a keyword).
func Lookup(ident []byte) int {
	// TODO should not have to convert every ident into a string
	//      for lookup - but at the moment maps of []byte don't
	//      seem to work - gri 3/3/09
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
