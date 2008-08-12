// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Scanner

import Platform "platform"
import Utils "utils"


export const (
	ILLEGAL = iota;
	EOF;
	IDENT;
	INT;
	FLOAT;
	STRING;

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
	
	SEND;
	RECV;

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
var VerboseMsgs bool;  // error message customization


export func TokenName(tok int) string {
	switch (tok) {
	case ILLEGAL: return "illegal";
	case EOF: return "eof";
	case IDENT: return "ident";
	case INT: return "int";
	case FLOAT: return "float";
	case STRING: return "string";

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
	
	case SEND: return "-<";
	case RECV: return "<-";

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


func init() {
	Keywords = new(map [string] int);
	
	for i := KEYWORDS_BEG; i <= KEYWORDS_END; i++ {
	  Keywords[TokenName(i)] = i;
	}
	
	// Provide column information in error messages for gri only...
	VerboseMsgs = Platform.USER == "gri";
}


func is_whitespace(ch int) bool {
	return ch == ' ' || ch == '\r' || ch == '\n' || ch == '\t';
}


func is_letter(ch int) bool {
	return 'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' || ch == '_' || ch >= 128 ;
}


func digit_val(ch int) int {
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


export type Scanner struct {
	filename string;  // error reporting only
	nerrors int;  // number of errors
	errpos int;  // last error position
	
	src string;  // scanned source
	pos int;  // current reading position
	ch int;  // one char look-ahead
	chpos int;  // position of ch
}


// Read the next Unicode char into S.ch.
// S.ch < 0 means end-of-file.
//
func (S *Scanner) Next() {
	const (
		Bit1 = 7;
		Bitx = 6;
		Bit2 = 5;
		Bit3 = 4;
		Bit4 = 3;

		T1 = (1 << (Bit1 + 1) - 1) ^ 0xFF;  // 0000 0000
		Tx = (1 << (Bitx + 1) - 1) ^ 0xFF;  // 1000 0000
		T2 = (1 << (Bit2 + 1) - 1) ^ 0xFF;  // 1100 0000
		T3 = (1 << (Bit3 + 1) - 1) ^ 0xFF;  // 1110 0000
		T4 = (1 << (Bit4 + 1) - 1) ^ 0xFF;  // 1111 0000

		Rune1 = 1 << (Bit1 + 0*Bitx) - 1;  // 0000 0000 0111 1111
		Rune2 = 1 << (Bit2 + 1*Bitx) - 1;  // 0000 0111 1111 1111
		Rune3 = 1 << (Bit3 + 2*Bitx) - 1;  // 1111 1111 1111 1111

		Maskx = 0x3F;  // 1 << Bitx - 1;  // 0011 1111
		Testx = 0xC0;  // Maskx ^ 0xFF;  // 1100 0000

		Bad	= 0xFFFD;  // Runeerror
	);

	src := S.src;
	lim := len(src);
	pos := S.pos;
	
	// 1-byte sequence
	// 0000-007F => T1
	if pos >= lim {
		S.ch = -1;  // end of file
		S.chpos = lim;
		return;
	}
	c0 := int(src[pos]);
	pos++;
	if c0 < Tx {
		S.ch = c0;
		S.chpos = S.pos;
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
		S.chpos = S.pos;
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
		S.chpos = S.pos;
		S.pos = pos;
		return;
	}

	// bad encoding
bad:
	S.ch = Bad;
	S.chpos = S.pos;
	S.pos += 1;
	return;
}


// Compute (line, column) information for a given source position.
func (S *Scanner) LineCol(pos int) (line, col int) {
	line = 1;
	lpos := 0;
	
	src := S.src;
	if pos > len(src) {
		pos = len(src);
	}

	for i := 0; i < pos; i++ {
		if src[i] == '\n' {
			line++;
			lpos = i;
		}
	}
	
	return line, pos - lpos;
}


func (S *Scanner) Error(pos int, msg string) {
	const errdist = 10;
	delta := pos - S.errpos;  // may be negative!
	if delta < 0 {
		delta = -delta;
	}
	if delta > errdist || S.nerrors == 0 /* always report first error */ {
		print S.filename;
		if pos >= 0 {
			// print position
			line, col := S.LineCol(pos);
			if VerboseMsgs {
				print ":", line, ":", col;
			} else {
				print ":", line;
			}
		}
		print ": ", msg, "\n";
		S.nerrors++;
		S.errpos = pos;
	}
	
	if S.nerrors >= 10 {
		sys.exit(1);
	}
}


func (S *Scanner) Open(filename, src string) {
	S.filename = filename;
	S.nerrors = 0;
	S.errpos = 0;
	
	S.src = src;
	S.pos = 0;
	S.Next();
}


func CharString(ch int) string {
	s := string(ch);
	switch ch {
	case '\a': s = `\a`;
	case '\b': s = `\b`;
	case '\f': s = `\f`;
	case '\n': s = `\n`;
	case '\r': s = `\r`;
	case '\t': s = `\t`;
	case '\v': s = `\v`;
	case '\\': s = `\\`;
	case '\'': s = `\'`;
	}
	return "'" + s + "' (U+" + Utils.IntToString(ch, 16) + ")";
}


func (S *Scanner) Expect(ch int) {
	if S.ch != ch {
		S.Error(S.chpos, "expected " + CharString(ch) + ", found " + CharString(S.ch));
	}
	S.Next();  // make always progress
}


func (S *Scanner) SkipWhitespace() {
	for is_whitespace(S.ch) {
		S.Next();
	}
}


func (S *Scanner) SkipComment() {
	// '/' already consumed
	if S.ch == '/' {
		// comment
		S.Next();
		for S.ch != '\n' && S.ch >= 0 {
			S.Next();
		}
		
	} else {
		/* comment */
		pos := S.chpos - 1;
		S.Expect('*');
		for S.ch >= 0 {
			ch := S.ch;
			S.Next();
			if ch == '*' && S.ch == '/' {
				S.Next();
				return;
			}
		}
		S.Error(pos, "comment not terminated");
	}
}


func (S *Scanner) ScanIdentifier() (tok int, val string) {
	pos := S.chpos;
	for is_letter(S.ch) || digit_val(S.ch) < 10 {
		S.Next();
	}
	val = S.src[pos : S.chpos];
	
	var present bool;
	tok, present = Keywords[val];
	if !present {
		tok = IDENT;
	}
	
	return tok, val;
}


func (S *Scanner) ScanMantissa(base int) {
	for digit_val(S.ch) < base {
		S.Next();
	}
}


func (S *Scanner) ScanNumber(seen_decimal_point bool) (tok int, val string) {
	pos := S.chpos;
	tok = INT;
	
	if seen_decimal_point {
		tok = FLOAT;
		pos--;  // '.' is one byte
		S.ScanMantissa(10);
		goto exponent;
	}
	
	if S.ch == '0' {
		// int or float
		S.Next();
		if S.ch == 'x' || S.ch == 'X' {
			// hexadecimal int
			S.Next();
			S.ScanMantissa(16);
		} else {
			// octal int or float
			S.ScanMantissa(8);
			if digit_val(S.ch) < 10 || S.ch == '.' || S.ch == 'e' || S.ch == 'E' {
				// float
				tok = FLOAT;
				goto mantissa;
			}
			// octal int
		}
		goto exit;
	}
	
mantissa:
	// decimal int or float
	S.ScanMantissa(10);
	
	if S.ch == '.' {
		// float
		tok = FLOAT;
		S.Next();
		S.ScanMantissa(10)
	}
	
exponent:
	if S.ch == 'e' || S.ch == 'E' {
		// float
		tok = FLOAT;
		S.Next();
		if S.ch == '-' || S.ch == '+' {
			S.Next();
		}
		S.ScanMantissa(10);
	}
	
exit:
	return tok, S.src[pos : S.chpos];
}


func (S *Scanner) ScanDigits(n int, base int) {
	for digit_val(S.ch) < base {
		S.Next();
		n--;
	}
	if n > 0 {
		S.Error(S.chpos, "illegal char escape");
	}
}


func (S *Scanner) ScanEscape() string {
	// TODO: fix this routine
	
	ch := S.ch;
	pos := S.chpos;
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
		S.Error(pos, "illegal char escape");
	}
}


func (S *Scanner) ScanChar() string {
	// '\'' already consumed

	pos := S.chpos - 1;
	ch := S.ch;
	S.Next();
	if ch == '\\' {
		S.ScanEscape();
	}

	S.Expect('\'');
	return S.src[pos : S.chpos];
}


func (S *Scanner) ScanString() string {
	// '"' already consumed

	pos := S.chpos - 1;
	for S.ch != '"' {
		ch := S.ch;
		S.Next();
		if ch == '\n' || ch < 0 {
			S.Error(pos, "string not terminated");
			break;
		}
		if ch == '\\' {
			S.ScanEscape();
		}
	}
	
	S.Next();
	return S.src[pos : S.chpos];
}


func (S *Scanner) ScanRawString() string {
	// '`' already consumed

	pos := S.chpos - 1;
	for S.ch != '`' {
		ch := S.ch;
		S.Next();
		if ch == '\n' || ch < 0 {
			S.Error(pos, "string not terminated");
			break;
		}
	}

	S.Next();
	return S.src[pos : S.chpos];
}


func (S *Scanner) Select2(tok0, tok1 int) int {
	if S.ch == '=' {
		S.Next();
		return tok1;
	}
	return tok0;
}


func (S *Scanner) Select3(tok0, tok1, ch2, tok2 int) int {
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


func (S *Scanner) Select4(tok0, tok1, ch2, tok2, tok3 int) int {
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


func (S *Scanner) Scan() (tok, pos int, val string) {
	S.SkipWhitespace();
	
	ch := S.ch;
	tok = ILLEGAL;
	pos = S.chpos;
	
	switch {
	case is_letter(ch): tok, val = S.ScanIdentifier();
	case digit_val(ch) < 10: tok, val = S.ScanNumber(false);
	default:
		S.Next();  // always make progress
		switch ch {
		case -1: tok = EOF;
		case '"': tok, val = STRING, S.ScanString();
		case '\'': tok, val = INT, S.ScanChar();
		case '`': tok, val = STRING, S.ScanRawString();
		case ':': tok = S.Select2(COLON, DEFINE);
		case '.':
			if digit_val(S.ch) < 10 {
				tok, val = S.ScanNumber(true);
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
		case '-':
			if S.ch == '<' {
				S.Next();
				tok = SEND;
			} else {
				tok = S.Select3(SUB, SUB_ASSIGN, '-', DEC);
			}
		case '*': tok = S.Select2(MUL, MUL_ASSIGN);
		case '/':
			if S.ch == '/' || S.ch == '*' {
				S.SkipComment();
				// cannot simply return because of 6g bug
				tok, pos, val = S.Scan();
				return tok, pos, val;
			}
			tok = S.Select2(QUO, QUO_ASSIGN);
		case '%': tok = S.Select2(REM, REM_ASSIGN);
		case '^': tok = S.Select2(XOR, XOR_ASSIGN);
		case '<':
			if S.ch == '-' {
				S.Next();
				tok = RECV;
			} else {
				tok = S.Select4(LSS, LEQ, '<', SHL, SHL_ASSIGN);
			}
		case '>': tok = S.Select4(GTR, GEQ, '>', SHR, SHR_ASSIGN);
		case '=': tok = S.Select2(ASSIGN, EQL);
		case '!': tok = S.Select2(NOT, NEQ);
		case '&': tok = S.Select3(AND, AND_ASSIGN, '&', LAND);
		case '|': tok = S.Select3(OR, OR_ASSIGN, '|', LOR);
		default:
			S.Error(pos, "illegal character " + CharString(ch));
			tok = ILLEGAL;
		}
	}
	
	return tok, pos, val;
}


export type Token struct {
	pos int;
	tok int;
	val string;
}


func (S *Scanner) Server(c *chan *Token) {
	for {
		t := new(Token);
		t.tok, t.pos, t.val = S.Scan();
		c -< t;
		if t.tok == EOF {
			break;
		}
	}
}
