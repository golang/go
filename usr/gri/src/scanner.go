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
	KEYWORDS_BEG = iota;
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
	KEYWORDS_END = iota;
)


var (
	Keywords *map [string] int;
)


export TokenName
func TokenName(tok int) string {
	switch (tok) {
	case ILLEGAL: return "illegal";
	case EOF: return "eof";
	case IDENT: return "ident";
	case STRING: return "string";
	case NUMBER: return "number";

	case COMMA: return ",";
	case COLON: return ":";
	case SEMICOLON: return ";";
	case PERIOD: return ".";

	case LPAREN: return "(";
	case RPAREN: return ")";
	case LBRACK: return "[";
	case RBRACK: return "]";
	case LBRACE: return "{";
	case RBRACE: return "}";

	case ASSIGN: return "=";
	case DEFINE: return ":=";
	
	case INC: return "++";
	case DEC: return "--";
	case NOT: return "!";

	case AND: return "&";
	case OR: return "|";
	case XOR: return "^";
	
	case ADD: return "+";
	case SUB: return "-";
	case MUL: return "*";
	case QUO: return "/";
	case REM: return "%";
	
	case EQL: return "==";
	case NEQ: return "!=";
	case LSS: return "<";
	case LEQ: return "<=";
	case GTR: return ">";
	case GEQ: return ">=";

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

	case CAND: return "&&";
	case COR: return "||";

	case BREAK: return "break";
	case CASE: return "case";
	case CONST: return "const";
	case CONTINUE: return "continue";
	case DEFAULT: return "default";
	case ELSE: return "else";
	case EXPORT: return "export";
	case FALLTHROUGH: return "fallthrough";
	case FALSE: return "false";
	case FOR: return "for";
	case FUNC: return "func";
	case GO: return "go";
	case GOTO: return "goto";
	case IF: return "if";
	case IMPORT: return "import";
	case INTERFACE: return "interface";
	case MAP: return "map";
	case NEW: return "new";
	case NIL: return "nil";
	case PACKAGE: return "package";
	case RANGE: return "range";
	case RETURN: return "return";
	case SELECT: return "select";
	case STRUCT: return "struct";
	case SWITCH: return "switch";
	case TRUE: return "true";
	case TYPE: return "type";
	case VAR: return "var";
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
	const (
		Bit1 = 7;
		Bitx = 6;
		Bit2 = 5;
		Bit3 = 4;
		Bit4 = 3;

		T1 = 0x00;  // (1 << (Bit1 + 1) - 1) ^ 0xFF;  // 0000 0000
		Tx = 0x80;  // (1 << (Bitx + 1) - 1) ^ 0xFF;  // 1000 0000
		T2 = 0xC0;  // (1 << (Bit2 + 1) - 1) ^ 0xFF;  // 1100 0000
		T3 = 0xE0;  // (1 << (Bit3 + 1) - 1) ^ 0xFF;  // 1110 0000
		T4 = 0xF0;  // (1 << (Bit4 + 1) - 1) ^ 0xFF;  // 1111 0000

		Rune1 = 1 << (Bit1 + 0*Bitx) - 1;  // 0000 0000 0111 1111
		Rune2 = 1 << (Bit2 + 1*Bitx) - 1;  // 0000 0111 1111 1111
		Rune3 = 1 << (Bit3 + 2*Bitx) - 1;  // 1111 1111 1111 1111

		Maskx = 0x3F;  // 1 << Bitx - 1;  // 0011 1111
		Testx = 0xC0;  // Maskx ^ 0xFF;  // 1100 0000

		Bad	= 0xFFFD;  // Runeerror
	);

	src := S.src;  // TODO only needed because of 6g bug
	lim := len(src);
	pos := S.pos;
	
	// 1-byte sequence
	// 0000-007F => T1
	if pos >= lim {
		goto eof;
	}
	c0 := int(src[pos + 0]);
	if c0 < Tx {
		S.ch = c0;
		S.pos = pos + 1;
		return;
	}

	// 2-byte sequence
	// 0080-07FF => T2 Tx
	if pos + 1 >= lim {
		goto eof;
	}
	c1 := int(src[pos + 1]) ^ Tx;
	if  c1 & Testx != 0 {
		goto bad;
	}
	if c0 < T3 {
		if c0 < T2 {
			goto bad;
		}
		r := (c0 << Bitx | c1) & Rune2;
		if  r <= Rune1 {
			goto bad;
		}
		S.ch = r;
		S.pos = pos + 2;
		return;
	}

	// 3-byte encoding
	// 0800-FFFF => T3 Tx Tx
	if pos + 2 >= lim {
		goto eof;
	}
	c2 := int(src[pos + 2]) ^ Tx;
	if c2 & Testx != 0 {
		goto bad;
	}
	if c0 < T4 {
		r := (((c0 << Bitx | c1) << Bitx) | c2) & Rune3;
		if r <= Rune2 {
			goto bad;
		}
		S.ch = r;
		S.pos = pos + 3;
		return;
	}

	// bad encoding
bad:
	S.ch = Bad;
	S.pos += 1;
	return;
	
	// end of file
eof:
	S.ch = -1;
}


func Init () {
	Keywords = new(map [string] int);
	
	for i := KEYWORDS_BEG; i <= KEYWORDS_END; i++ {
	  Keywords[TokenName(i)] = i;
	}
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
