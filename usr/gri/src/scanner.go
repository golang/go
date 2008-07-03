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
	
	OR = iota;
	BOR = iota;
	AND = iota;
	BAND = iota;
	
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
	
	case OR: return "OR";
	case BOR: return "BOR";
	case AND: return "AND";
	case BAND: return "BAND";
	
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


func (S *Scanner) ScanNumber () {
	// TODO complete this routine
	
	for is_dec_digit(S.ch) {
		S.Next();
	}
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


func (S *Scanner) ScanChar () {
	S.Next();  // consume '\'

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
}


func (S *Scanner) ScanString () {
	for S.Next(); S.ch != '"'; S.Next() {
		if S.ch == '\n' || S.ch < 0 {
			panic "string not terminated";
		}
	}
	S.Next();
}


func (S *Scanner) ScanRawString () {
	for S.Next(); S.ch != '`'; S.Next() {
		if S.ch == '\n' || S.ch < 0 {
			panic "string not terminated";
		}
	}
	S.Next();
}


func (S *Scanner) Scan () (tok, beg, end int) {
	S.SkipWhitespace();
	
	var tok int = ILLEGAL;
	var beg int = S.pos - 1;
	var end int = beg;
	
	if is_letter(S.ch) {
		tok = S.ScanIdentifier();

	} else if is_dec_digit(S.ch) {
		S.ScanNumber();
		tok = NUMBER;

	} else {
		switch S.ch {
			case -1:
				tok = EOF;
				
			case '/':
				S.Next();
				if S.ch == '/' || S.ch == '*' {
					S.SkipComment();
					tok, beg, end = S.Scan();
					return tok, beg, end;
				} else {
					tok = QUO;
				}
				
			case '"':
				S.ScanString();
				tok = STRING;
				
			case '\'':
				S.ScanChar();
				tok = NUMBER;
				
			case '`':
				S.ScanRawString();
				tok = STRING;
				
			case ':':
				S.Next();
				if (S.ch == '=') {
					S.Next();
					tok = DEFINE;
				} else {
					tok = COLON;
				}
				
			case '.':
				S.Next();
				tok = PERIOD;
				
			case ',':
				S.Next();
				tok = COMMA;
				
			case '+':
				S.Next();
				if (S.ch == '+') {
					S.Next();
					tok = INC;
				} else {
					tok = ADD;
				}
				
			case '-':
				S.Next();
				if (S.ch == '-') {
					S.Next();
					tok = DEC;
				} else {
					tok = SUB;
				}
				
			case '*':
				S.Next();
				tok = MUL;

			case '/':
				S.Next();
				tok = QUO;

			case '%':
				S.Next();
				tok = REM;

			case '<':
				S.Next();
				if (S.ch == '=') {
					S.Next();
					tok = LEQ;
				} else {
					tok = LSS;
				}
				
			case '>':
				S.Next();
				if (S.ch == '=') {
					S.Next();
					tok = GEQ;
				} else {
					tok = GTR;
				}
				
			case '=':
				S.Next();
				if (S.ch == '=') {
					S.Next();
					tok = EQL;
				} else {
					tok = ASSIGN;
				}
				
			case '!':
				S.Next();
				if (S.ch == '=') {
					S.Next();
					tok = NEQ;
				} else {
					tok = NOT;
				}
				
			case ';':
				S.Next();
				tok = SEMICOLON;
				
			case '(':
				S.Next();
				tok = LPAREN;
				
			case ')':
				S.Next();
				tok = LPAREN;
				
			case '[':
				S.Next();
				tok = LBRACK;
				
			case ']':
				S.Next();
				tok = RBRACK;
				
			case '{':
				S.Next();
				tok = LBRACE;
				
			case '}':
				S.Next();
				tok = RBRACE;
				
			case '&':
				S.Next();
				if S.ch == '&' {
					S.Next();
					tok = AND;
				} else {
					tok = BAND;
				}
				
			case '|':
				S.Next();
				if S.ch == '|' {
					S.Next();
					tok = OR;
				} else {
					tok = BOR;
				}
				
			default:
				S.Next();  // make progress
		}
	}
	
	end = S.pos - 1;
	return tok, beg, end;
}
