// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Scanner


export EOF;
const (
	ILLEGAL = iota;
	EOF = iota;
	IDENT = iota;
	STRING = iota;
	NUMBER = iota;

	COMMA = iota;
	COLON = iota;
	SEMICOLON = iota;
	PERIOD = iota;

	LPAREN = iota;
	RPAREN = iota;
	LBRACK = iota;
	RBRACK = iota;
	LBRACE = iota;
	RBRACE = iota;
	
	ASSIGN = iota;
	DEFINE = iota;
	
	INC = iota;
	DEC = iota;
	NOT = iota;
	
	AND = iota;
	OR = iota;
	XOR = iota;
	
	ADD = iota;
	SUB = iota;
	MUL = iota;
	QUO = iota;
	REM = iota;
	
	EQL = iota;
	NEQ = iota;
	LSS = iota;
	LEQ = iota;
	GTR = iota;
	GEQ = iota;

	SHL = iota;
	SHR = iota;

	ADD_ASSIGN = iota;
	SUB_ASSIGN = iota;
	MUL_ASSIGN = iota;
	QUO_ASSIGN = iota;
	REM_ASSIGN = iota;

	AND_ASSIGN = iota;
	OR_ASSIGN = iota;
	XOR_ASSIGN = iota;
	
	SHL_ASSIGN = iota;
	SHR_ASSIGN = iota;

	CAND = iota;
	COR = iota;
	
	// keywords
	BREAK = iota;
	CASE = iota;
	CONST = iota;
	CONTINUE = iota;
	DEFAULT = iota;
	ELSE = iota;
	EXPORT = iota;
	FALLTHROUGH = iota;
	FALSE = iota;
	FOR = iota;
	FUNC = iota;
	GO = iota;
	GOTO = iota;
	IF = iota;
	IMPORT = iota;
	INTERFACE = iota;
	MAP = iota;
	NEW = iota;
	NIL = iota;
	PACKAGE = iota;
	RANGE = iota;
	RETURN = iota;
	SELECT = iota;
	STRUCT = iota;
	SWITCH = iota;
	TRUE = iota;
	TYPE = iota;
	VAR = iota;
)


var (
	Keywords *map [string] int;
)


export TokenName
func TokenName(tok int) string {
	switch (tok) {
	case ILLEGAL: return "ILLEGAL";
	case EOF: return "EOF";
	case IDENT: return "IDENT";
	case STRING: return "STRING";
	case NUMBER: return "NUMBER";

	case COMMA: return "COMMA";
	case COLON: return "COLON";
	case SEMICOLON: return "SEMICOLON";
	case PERIOD: return "PERIOD";

	case LPAREN: return "LPAREN";
	case RPAREN: return "RPAREN";
	case LBRACK: return "LBRACK";
	case RBRACK: return "RBRACK";
	case LBRACE: return "LBRACE";
	case RBRACE: return "RBRACE";

	case ASSIGN: return "ASSIGN";
	case DEFINE: return "DEFINE";
	
	case INC: return "INC";
	case DEC: return "DEC";
	case NOT: return "NOT";

	case AND: return "AND";
	case OR: return "OR";
	case XOR: return "XOR";
	
	case ADD: return "ADD";
	case SUB: return "SUB";
	case MUL: return "MUL";
	case REM: return "REM";
	case QUO: return "QUO";
	case REM: return "REM";
	
	case EQL: return "EQL";
	case NEQ: return "NEQ";
	case LSS: return "LSS";
	case LEQ: return "LEQ";
	case GTR: return "GTR";
	case GEQ: return "GEQ";

	case SHL: return SHL;
	case SHR: return SHR;

	case ADD_ASSIGN: return "ADD_ASSIGN";
	case SUB_ASSIGN: return "SUB_ASSIGN";
	case MUL_ASSIGN: return "MUL_ASSIGN";
	case QUO_ASSIGN: return "QUO_ASSIGN";
	case REM_ASSIGN: return "REM_ASSIGN";

	case AND_ASSIGN: return "AND_ASSIGN";
	case OR_ASSIGN: return "OR_ASSIGN";
	case XOR_ASSIGN: return "XOR_ASSIGN";

	case SHL_ASSIGN: return "SHL_ASSIGN";
	case SHR_ASSIGN: return "SHR_ASSIGN";

	case CAND: return "CAND";
	case COR: return "COR";

	case BREAK: return "BREAK";
	case CASE: return "CASE";
	case CONST: return "CONST";
	case CONTINUE: return "CONTINUE";
	case DEFAULT: return "DEFAULT";
	case ELSE: return "ELSE";
	case EXPORT: return "EXPORT";
	case FALLTHROUGH: return "FALLTHROUGH";
	case FALSE: return "FALSE";
	case FOR: return "FOR";
	case FUNC: return "FUNC";
	case GO: return "GO";
	case GOTO: return "GOTO";
	case IF: return "IF";
	case IMPORT: return "IMPORT";
	case INTERFACE: return "INTERFACE";
	case MAP: return "MAP";
	case NEW: return "NEW";
	case NIL: return "NIL";
	case PACKAGE: return "PACKAGE";
	case RANGE: return "RANGE";
	case RETURN: return "RETURN";
	case SELECT: return "SELECT";
	case STRUCT: return "STRUCT";
	case SWITCH: return "SWITCH";
	case TRUE: return "TRUE";
	case TYPE: return "TYPE";
	case VAR: return "VAR";
	}
	
	return "???";
}


func is_whitespace (ch int) bool {
	return ch == ' ' || ch == '\r' || ch == '\n' || ch == '\t';
}


func is_letter (ch int) bool {
	return 'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' || ch == '_' || ch >= 128 ;
}


func is_oct_digit (ch int) bool {
	return '0' <= ch && ch <= '7';
}


func is_dec_digit (ch int) bool {
	return '0' <= ch && ch <= '9';
}


func is_hex_digit (ch int) bool {
	return '0' <= ch && ch <= '9' || 'a' <= ch && ch <= 'f' || 'A' <= ch && ch <= 'F';
}


export Scanner
type Scanner struct {
	src string;
	pos int;
	ch int;  // one char look-ahead
}


func (S *Scanner) Next () {
	src := S.src;  // TODO only needed because of 6g bug
	if S.pos < len(src) {
		S.ch = int(S.src[S.pos]);
		S.pos++;
		if (S.ch >= 128) {
			panic "UTF-8 not handled"
		}
	} else {
		S.ch = -1;
	}
}


func Init () {
	Keywords = new(map [string] int);

	Keywords["break"] = BREAK;
	Keywords["case"] = CASE;
	Keywords["const"] = CONST;
	Keywords["continue"] = CONTINUE;
	Keywords["default"] = DEFAULT;
	Keywords["else"] = ELSE;
	Keywords["export"] = EXPORT;
	Keywords["fallthrough"] = FALLTHROUGH;
	Keywords["false"] = FALSE;
	Keywords["for"] = FOR;
	Keywords["func"] = FUNC;
	Keywords["go"] = GO;
	Keywords["goto"] = GOTO;
	Keywords["if"] = IF;
	Keywords["import"] = IMPORT;
	Keywords["interface"] = INTERFACE;
	Keywords["map"] = MAP;
	Keywords["new"] = NEW;
	Keywords["nil"] = NIL;
	Keywords["package"] = PACKAGE;
	Keywords["range"] = RANGE;
	Keywords["return"] = RETURN;
	Keywords["select"] = SELECT;
	Keywords["struct"] = STRUCT;
	Keywords["switch"] = SWITCH;
	Keywords["true"] = TRUE;
	Keywords["type"] = TYPE;
	Keywords["var"] = VAR;
}


func (S *Scanner) Open (src string) {
	if Keywords == nil {
		Init();
	}

	S.src = src;
	S.pos = 0;
	S.Next();
}


func (S *Scanner) SkipWhitespace () {
	for is_whitespace(S.ch) {
		S.Next();
	}
}


func (S *Scanner) SkipComment () {
	if S.ch == '/' {
		// comment
		for S.Next(); S.ch != '\n' && S.ch >= 0; S.Next() {}
		
	} else {
		/* comment */
		for S.Next(); S.ch >= 0; {
			c := S.ch;
			S.Next();
			if c == '*' && S.ch == '/' {
				S.Next();
				return;
			}
		}
		panic "comment not terminated";
	}
}


func (S *Scanner) ScanIdentifier () int {
	beg := S.pos - 1;
	for is_letter(S.ch) || is_dec_digit(S.ch) {
		S.Next();
	}
	end := S.pos - 1;
	
	var tok int;
	var present bool;
	tok, present = Keywords[S.src[beg : end]];
	if !present {
		tok = IDENT;
	}
	
	return tok;
}


func (S *Scanner) ScanMantissa () {
     	for is_dec_digit(S.ch) {
	      	S.Next();
	}
}


func (S *Scanner) ScanNumber () int {
	// TODO complete this routine
	if S.ch == '.' {
	   	S.Next();
	}
	S.ScanMantissa();
	if S.ch == 'e' || S.ch == 'E' {
	   	S.Next();
		if S.ch == '-' || S.ch == '+' {
		   	S.Next();
		}
		S.ScanMantissa();
	}
	return NUMBER;
}


func (S *Scanner) ScanOctDigits(n int) {
	for ; n > 0; n-- {
		if !is_oct_digit(S.ch) {
			panic "illegal char escape";
		}
		S.Next();
	}
}


func (S *Scanner) ScanHexDigits(n int) {
	for ; n > 0; n-- {
		if !is_hex_digit(S.ch) {
			panic "illegal char escape";
		}
		S.Next();
	}
}


func (S *Scanner) ScanEscape () {
	// TODO: fix this routine
	
	switch (S.ch) {
	case 'a': fallthrough;
	case 'b': fallthrough;
	case 'f': fallthrough;
	case 'n': fallthrough;
	case 'r': fallthrough;
	case 't': fallthrough;
	case 'v': fallthrough;
	case '\\': fallthrough;
	case '\'': fallthrough;
	case '"':
		S.Next();
		
	case '0', '1', '2', '3', '4', '5', '6', '7':
		S.ScanOctDigits(3);
		
	case 'x':
		S.Next();
		S.ScanHexDigits(2);
		
	case 'u':
		S.Next();
		S.ScanHexDigits(4);

	case 'U':
		S.Next();
		S.ScanHexDigits(8);

	default:
		panic "illegal char escape";
	}
}


func (S *Scanner) ScanChar () int {
	if (S.ch == '\\') {
		S.Next();
		S.ScanEscape();
	} else {
		S.Next();
	}

	if S.ch == '\'' {
		S.Next();
	} else {
		panic "char not terminated";
	}
	return NUMBER;
}


func (S *Scanner) ScanString () int {
	for ; S.ch != '"'; S.Next() {
		if S.ch == '\n' || S.ch < 0 {
			panic "string not terminated";
		}
	}
	S.Next();
	return STRING;
}


func (S *Scanner) ScanRawString () int {
	for ; S.ch != '`'; S.Next() {
		if S.ch == '\n' || S.ch < 0 {
			panic "string not terminated";
		}
	}
	S.Next();
	return STRING;
}


func (S *Scanner) Select2 (tok0, tok1 int) int {
     	if S.ch == '=' {
	   	S.Next();
	   	return tok1;
	}
	return tok0;
}


func (S *Scanner) Select3 (tok0, tok1, ch2, tok2 int) int {
	if S.ch == '=' {
	   	S.Next();
		return tok1;
	}
	if S.ch == ch2 {
	   	S.Next();
		return tok2;
	}
	return tok0;
}


func (S *Scanner) Select4 (tok0, tok1, ch2, tok2, tok3 int) int {
	if S.ch == '=' {
	   	S.Next();
		return tok1;
	}
	if S.ch == ch2 {
	   	S.Next();
		if S.ch == '=' {
		   	S.Next();
			return tok3;
		}
		return tok2;
	}
	return tok0;
}


func (S *Scanner) Scan () (tok, beg, end int) {
	S.SkipWhitespace();
	
	var tok int = ILLEGAL;
	var beg int = S.pos - 1;
	var end int = beg;
	
	switch ch := S.ch; {
	case is_letter(ch): tok = S.ScanIdentifier();
	case is_dec_digit(ch): tok = S.ScanNumber();
	default:
		S.Next();
		switch ch {
			case -1: tok = EOF;
			case '"': tok = S.ScanString();
			case '\'': tok = S.ScanChar();
			case '`': tok = S.ScanRawString();
			case ':': tok = S.Select2(COLON, DEFINE);
			case '.':
				if is_dec_digit(S.ch) {
				     	tok = S.ScanNumber();
				} else {
				        tok = PERIOD;
				}
			case ',': tok = COMMA;
			case ';': tok = SEMICOLON;
			case '(': tok = LPAREN;
			case ')': tok = RPAREN;
			case '[': tok = LBRACK;
			case ']': tok = RBRACK;
			case '{': tok = LBRACE;
			case '}': tok = RBRACE;
			case '+': tok = S.Select3(ADD, ADD_ASSIGN, '+', INC);
			case '-': tok = S.Select3(SUB, SUB_ASSIGN, '-', DEC);
			case '*': tok = S.Select2(MUL, MUL_ASSIGN);
			case '/':
				if S.ch == '/' || S.ch == '*' {
					S.SkipComment();
					// cannot simply return because of 6g bug
					tok, beg, end = S.Scan();
					return tok, beg, end;
				}
				tok = S.Select2(QUO, QUO_ASSIGN);
			case '%': tok = S.Select2(REM, REM_ASSIGN);
			case '^': tok = S.Select2(XOR, XOR_ASSIGN);
			case '<': tok = S.Select4(LSS, LEQ, '<', SHL, SHL_ASSIGN);
			case '>': tok = S.Select4(GTR, GEQ, '>', SHR, SHR_ASSIGN);
			case '=': tok = S.Select2(ASSIGN, EQL);
			case '!': tok = S.Select2(NOT, NEQ);
			case '&': tok = S.Select3(AND, AND_ASSIGN, '&', CAND);
			case '|': tok = S.Select3(OR, OR_ASSIGN, '|', COR);
			default: tok = ILLEGAL;

		}
	}
	
	end = S.pos - 1;
	return tok, beg, end;
}
