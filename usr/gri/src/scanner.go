// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Scanner

export
	ILLEGAL, EOF, IDENT, STRING, NUMBER,
	COMMA, COLON, SEMICOLON, PERIOD,
	LPAREN, RPAREN, LBRACK, RBRACK, LBRACE, RBRACE,
	ASSIGN, DEFINE,
	INC, DEC, NOT,
	AND, OR, XOR,
	ADD, SUB, MUL, QUO, REM,
	EQL, NEQ, LSS, LEQ, GTR, GEQ,
	SHL, SHR,
	ADD_ASSIGN, SUB_ASSIGN, MUL_ASSIGN, QUO_ASSIGN, REM_ASSIGN,
	AND_ASSIGN, OR_ASSIGN, XOR_ASSIGN, SHL_ASSIGN, SHR_ASSIGN,
	CAND, COR,
	BREAK, CASE, CHAN, CONST, CONTINUE, DEFAULT, ELSE, EXPORT, FALLTHROUGH, FALSE,
	FOR, FUNC, GO, GOTO, IF, IMPORT, INTERFACE, IOTA, MAP, NEW, NIL, PACKAGE, RANGE,
	RETURN, SELECT, STRUCT, SWITCH, TRUE, TYPE, VAR
	
	
const (
	ILLEGAL = iota;
	EOF;
	IDENT;
	STRING;
	NUMBER;

	COMMA;
	COLON;
	SEMICOLON;
	PERIOD;

	LPAREN;
	RPAREN;
	LBRACK;
	RBRACK;
	LBRACE;
	RBRACE;
	
	ASSIGN;
	DEFINE;
	
	INC;
	DEC;
	NOT;
	
	AND;
	OR;
	XOR;
	
	ADD;
	SUB;
	MUL;
	QUO;
	REM;
	
	EQL;
	NEQ;
	LSS;
	LEQ;
	GTR;
	GEQ;

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

	CAND;
	COR;
	
	// keywords
	KEYWORDS_BEG;
	BREAK;
	CASE;
	CHAN;
	CONST;
	CONTINUE;
	DEFAULT;
	ELSE;
	EXPORT;
	FALLTHROUGH;
	FALSE;
	FOR;
	FUNC;
	GO;
	GOTO;
	IF;
	IMPORT;
	INTERFACE;
	IOTA;
	MAP;
	NEW;
	NIL;
	PACKAGE;
	RANGE;
	RETURN;
	SELECT;
	STRUCT;
	SWITCH;
	TRUE;
	TYPE;
	VAR;
	KEYWORDS_END;
)


var Keywords *map [string] int;


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
	case LBRACE: return "LBRACE";
	case RBRACE: return "RBRACE";

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
	case CHAN: return "chan";
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
	case IOTA: return "iota";
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


func digit_val (ch int) int {
	if '0' <= ch && ch <= '9' {
		return ch - '0';
	}
	if 'a' <= ch && ch <= 'f' {
		return ch - 'a' + 10;
	}
	if 'A' <= ch && ch <= 'F' {
		return ch - 'A' + 10;
	}
	return 16;  // larger than any legal digit val
}


export Scanner
type Scanner struct {
	src string;
	pos int;
	ch int;  // one char look-ahead
}


/*
export Token
type Token struct {
	val int;
	beg, end int;
	txt string;
}


func (T *Token) Print () {
	print TokenName(T.val), " [", T.beg, ", ", T.end, "[ ", T.txt, "\n";
}
*/


// Read the next Unicode char into S.ch.
// S.ch < 0 means end-of-file.
//
func (S *Scanner) Next () {
	const (
		Bit1 = 7;
		Bitx = 6;
		Bit2 = 5;
		Bit3 = 4;
		Bit4 = 3;

		// TODO 6g constant evaluation incomplete
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
		S.ch = -1;  // end of file
		return;
	}
	c0 := int(src[pos]);
	pos++;
	if c0 < Tx {
		S.ch = c0;
		S.pos = pos;
		return;
	}

	// 2-byte sequence
	// 0080-07FF => T2 Tx
	if pos >= lim {
		goto bad;
	}
	c1 := int(src[pos]) ^ Tx;
	pos++;
	if c1 & Testx != 0 {
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
		S.pos = pos;
		return;
	}

	// 3-byte sequence
	// 0800-FFFF => T3 Tx Tx
	if pos >= lim {
		goto bad;
	}
	c2 := int(src[pos]) ^ Tx;
	pos++;
	if c2 & Testx != 0 {
		goto bad;
	}
	if c0 < T4 {
		r := (((c0 << Bitx | c1) << Bitx) | c2) & Rune3;
		if r <= Rune2 {
			goto bad;
		}
		S.ch = r;
		S.pos = pos;
		return;
	}

	// bad encoding
bad:
	S.ch = Bad;
	S.pos += 1;
	return;
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


func (S *Scanner) Expect (ch int) {
	if S.ch != ch {
		panic "expected ", string(ch), " found ", string(S.ch);
	}
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
		S.Next();
		for S.ch != '\n' && S.ch >= 0 {
			S.Next();
		}
		
	} else {
		/* comment */
		S.Next();
		for S.ch >= 0 {
			ch := S.ch;
			S.Next();
			if ch == '*' && S.ch == '/' {
				S.Next();
				return;
			}
		}
		panic "comment not terminated";
	}
}


func (S *Scanner) ScanIdentifier () int {
	beg := S.pos - 1;
	for is_letter(S.ch) || digit_val(S.ch) < 10 {
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


func (S *Scanner) ScanMantissa (base int) {
	for digit_val(S.ch) < base {
		S.Next();
	}
}


func (S *Scanner) ScanNumber (seen_decimal_point bool) int {
	if seen_decimal_point {
		S.ScanMantissa(10);
		goto exponent;
	}
	
	if S.ch == '0' {
		// TODO bug: doesn't accept 09.0 !
		// int
		S.Next();
		if S.ch == 'x' || S.ch == 'X' {
			// hexadecimal int
			S.Next();
			S.ScanMantissa(16);
		} else {
			// octal int
			S.ScanMantissa(8);
		}
		return NUMBER;
	}
	
	// decimal int or float
	S.ScanMantissa(10);
	
	if S.ch == '.' {
		// float
		S.Next();
		S.ScanMantissa(10)
	}
	
exponent:
	if S.ch == 'e' || S.ch == 'E' {
		// float
		S.Next();
		if S.ch == '-' || S.ch == '+' {
			S.Next();
		}
		S.ScanMantissa(10);
	}
	return NUMBER;
}


func (S *Scanner) ScanDigits(n int, base int) {
	for digit_val(S.ch) < base {
		S.Next();
		n--;
	}
	if n > 0 {
		panic "illegal char escape";
	}
}


func (S *Scanner) ScanEscape () string {
	// TODO: fix this routine
	
	ch := S.ch;
	S.Next();
	switch (ch) {
	case 'a', 'b', 'f', 'n', 'r', 't', 'v', '\\', '\'', '"':
		return string(ch);
		
	case '0', '1', '2', '3', '4', '5', '6', '7':
		S.ScanDigits(3 - 1, 8);  // 1 char already read
		return "";  // TODO fix this
		
	case 'x':
		S.ScanDigits(2, 16);
		return "";  // TODO fix this
		
	case 'u':
		S.ScanDigits(4, 16);
		return "";  // TODO fix this

	case 'U':
		S.ScanDigits(8, 16);
		return "";  // TODO fix this

	default:
		panic "illegal char escape";
	}
}


func (S *Scanner) ScanChar () int {
	// '\'' already consumed

	ch := S.ch;
	S.Next();
	if ch == '\\' {
		S.ScanEscape();
	}

	S.Expect('\'');
	return NUMBER;
}


func (S *Scanner) ScanString () int {
	// '"' already consumed

	for S.ch != '"' {
		ch := S.ch;
		S.Next();
		if ch == '\n' || ch < 0 {
			panic "string not terminated";
		}
		if ch == '\\' {
			S.ScanEscape();
		}
	}

	S.Next();
	return STRING;
}


func (S *Scanner) ScanRawString () int {
	// '`' already consumed

	for S.ch != '`' {
		ch := S.ch;
		S.Next();
		if ch == '\n' || ch < 0 {
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
	
	tok = ILLEGAL;
	beg = S.pos - 1;
	end = beg;
	
	ch := S.ch;
	switch {
	case is_letter(ch): tok = S.ScanIdentifier();
	case digit_val(ch) < 10: tok = S.ScanNumber(false);
	default:
		S.Next();
		switch ch {
		case -1: tok = EOF;
		case '"': tok = S.ScanString();
		case '\'': tok = S.ScanChar();
		case '`': tok = S.ScanRawString();
		case ':': tok = S.Select2(COLON, DEFINE);
		case '.':
			if digit_val(S.ch) < 10 {
				tok = S.ScanNumber(true);
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
	
	/*
	t.val = tok;
	t.beg = beg;
	t.end = end;
	t.txt = S.src[beg : end];
	*/
	
	return tok, beg, end;
}
