// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scanner

import (
	"utf8";
	"unicode";
	"strconv";
)

const (
	ILLEGAL = iota;
	EOF;
	
	INT;
	FLOAT;
	STRING;
	IDENT;
	COMMENT;

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

	// keywords
	keywords_beg;
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
	keywords_end;
)


func TokenString(tok int) string {
	switch tok {
	case ILLEGAL: return "ILLEGAL";
	case EOF: return "EOF";

	case INT: return "INT";
	case FLOAT: return "FLOAT";
	case STRING: return "STRING";
	case IDENT: return "IDENT";
	case COMMENT: return "COMMENT";

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
	for i := keywords_beg + 1; i < keywords_end; i++ {
		keywords[TokenString(i)] = i;
	}
}


func is_letter(ch int) bool {
	return
		'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' ||  // common case
		ch == '_' || unicode.IsLetter(ch);
}


func digit_val(ch int) int {
	// TODO: spec permits other Unicode digits as well
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


type ErrorHandler interface {
	Error(pos int, msg string);
}


type Scanner struct {
	// setup
	src []byte;  // source
	err ErrorHandler;
	scan_comments bool;

	// scanning
	pos int;  // current reading position
	ch int;  // one char look-ahead
	chpos int;  // position of ch
}


// Read the next Unicode char into S.ch.
// S.ch < 0 means end-of-file.
func (S *Scanner) next() {
	if S.pos < len(S.src) {
		// assume ascii
		r, w := int(S.src[S.pos]), 1;
		if r >= 0x80 {
			// not ascii
			r, w = utf8.DecodeRune(S.src[S.pos : len(S.src)]);
		}
		S.ch = r;
		S.chpos = S.pos;
		S.pos += w;
	} else {
		S.ch = -1;  // eof
		S.chpos = len(S.src);
	}
}


func (S *Scanner) error(pos int, msg string) {
	S.err.Error(pos, msg);
}


func (S *Scanner) Init(src []byte, err ErrorHandler, scan_comments bool) {
	S.src = src;
	S.err = err;
	S.scan_comments = scan_comments;
	S.next();
}


func charString(ch int) string {
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
	return "'" + s + "' (U+" + strconv.Itob(ch, 16) + ")";
}


func (S *Scanner) expect(ch int) {
	if S.ch != ch {
		S.error(S.chpos, "expected " + charString(ch) + ", found " + charString(S.ch));
	}
	S.next();  // make always progress
}


func (S *Scanner) skipWhitespace() {
	for {
		switch S.ch {
		case '\t', '\r', ' ':
			// nothing to do
		case '\n':
			if S.scan_comments {
				return;
			}
		default:
			return;
		}
		S.next();
	}
	panic("UNREACHABLE");
}


func (S *Scanner) scanComment() []byte {
	// first '/' already consumed
	pos := S.chpos - 1;

	if S.ch == '/' {
		//-style comment
		for S.ch >= 0 {
			S.next();
			if S.ch == '\n' {
				// '\n' terminates comment but we do not include
				// it in the comment (otherwise we don't see the
				// start of a newline in skipWhitespace()).
				goto exit;
			}
		}

	} else {
		/*-style comment */
		S.expect('*');
		for S.ch >= 0 {
			ch := S.ch;
			S.next();
			if ch == '*' && S.ch == '/' {
				S.next();
				goto exit;
			}
		}
	}

	S.error(pos, "comment not terminated");

exit:
	return S.src[pos : S.chpos];
}


func (S *Scanner) scanIdentifier() (tok int, val []byte) {
	pos := S.chpos;
	for is_letter(S.ch) || digit_val(S.ch) < 10 {
		S.next();
	}
	val = S.src[pos : S.chpos];

	var present bool;
	tok, present = keywords[string(val)];
	if !present {
		tok = IDENT;
	}

	return tok, val;
}


func (S *Scanner) scanMantissa(base int) {
	for digit_val(S.ch) < base {
		S.next();
	}
}


func (S *Scanner) scanNumber(seen_decimal_point bool) (tok int, val []byte) {
	pos := S.chpos;
	tok = INT;

	if seen_decimal_point {
		tok = FLOAT;
		pos--;  // '.' is one byte
		S.scanMantissa(10);
		goto exponent;
	}

	if S.ch == '0' {
		// int or float
		S.next();
		if S.ch == 'x' || S.ch == 'X' {
			// hexadecimal int
			S.next();
			S.scanMantissa(16);
		} else {
			// octal int or float
			S.scanMantissa(8);
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
	S.scanMantissa(10);

	if S.ch == '.' {
		// float
		tok = FLOAT;
		S.next();
		S.scanMantissa(10)
	}

exponent:
	if S.ch == 'e' || S.ch == 'E' {
		// float
		tok = FLOAT;
		S.next();
		if S.ch == '-' || S.ch == '+' {
			S.next();
		}
		S.scanMantissa(10);
	}

exit:
	return tok, S.src[pos : S.chpos];
}


func (S *Scanner) scanDigits(n int, base int) {
	for digit_val(S.ch) < base {
		S.next();
		n--;
	}
	if n > 0 {
		S.error(S.chpos, "illegal char escape");
	}
}


func (S *Scanner) scanEscape(quote int) {
	ch := S.ch;
	pos := S.chpos;
	S.next();
	switch ch {
	case 'a', 'b', 'f', 'n', 'r', 't', 'v', '\\', quote:
		// nothing to do
	case '0', '1', '2', '3', '4', '5', '6', '7':
		S.scanDigits(3 - 1, 8);  // 1 char read already
	case 'x':
		S.scanDigits(2, 16);
	case 'u':
		S.scanDigits(4, 16);
	case 'U':
		S.scanDigits(8, 16);
	default:
		S.error(pos, "illegal char escape");
	}
}


func (S *Scanner) scanChar() []byte {
	// '\'' already consumed

	pos := S.chpos - 1;
	ch := S.ch;
	S.next();
	if ch == '\\' {
		S.scanEscape('\'');
	}

	S.expect('\'');
	return S.src[pos : S.chpos];
}


func (S *Scanner) scanString() []byte {
	// '"' already consumed

	pos := S.chpos - 1;
	for S.ch != '"' {
		ch := S.ch;
		S.next();
		if ch == '\n' || ch < 0 {
			S.error(pos, "string not terminated");
			break;
		}
		if ch == '\\' {
			S.scanEscape('"');
		}
	}

	S.next();
	return S.src[pos : S.chpos];
}


func (S *Scanner) scanRawString() []byte {
	// '`' already consumed

	pos := S.chpos - 1;
	for S.ch != '`' {
		ch := S.ch;
		S.next();
		if ch == '\n' || ch < 0 {
			S.error(pos, "string not terminated");
			break;
		}
	}

	S.next();
	return S.src[pos : S.chpos];
}


func (S *Scanner) select2(tok0, tok1 int) int {
	if S.ch == '=' {
		S.next();
		return tok1;
	}
	return tok0;
}


func (S *Scanner) select3(tok0, tok1, ch2, tok2 int) int {
	if S.ch == '=' {
		S.next();
		return tok1;
	}
	if S.ch == ch2 {
		S.next();
		return tok2;
	}
	return tok0;
}


func (S *Scanner) select4(tok0, tok1, ch2, tok2, tok3 int) int {
	if S.ch == '=' {
		S.next();
		return tok1;
	}
	if S.ch == ch2 {
		S.next();
		if S.ch == '=' {
			S.next();
			return tok3;
		}
		return tok2;
	}
	return tok0;
}


func (S *Scanner) Scan() (pos, tok int, val []byte) {
loop:
	S.skipWhitespace();

	pos, tok = S.chpos, ILLEGAL;

	switch ch := S.ch; {
	case is_letter(ch): tok, val = S.scanIdentifier();
	case digit_val(ch) < 10: tok, val = S.scanNumber(false);
	default:
		S.next();  // always make progress
		switch ch {
		case -1: tok = EOF;
		case '\n': tok, val = COMMENT, []byte{'\n'};
		case '"': tok, val = STRING, S.scanString();
		case '\'': tok, val = INT, S.scanChar();
		case '`': tok, val = STRING, S.scanRawString();
		case ':': tok = S.select2(COLON, DEFINE);
		case '.':
			if digit_val(S.ch) < 10 {
				tok, val = S.scanNumber(true);
			} else if S.ch == '.' {
				S.next();
				if S.ch == '.' {
					S.next();
					tok = ELLIPSIS;
				}
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
		case '+': tok = S.select3(ADD, ADD_ASSIGN, '+', INC);
		case '-': tok = S.select3(SUB, SUB_ASSIGN, '-', DEC);
		case '*': tok = S.select2(MUL, MUL_ASSIGN);
		case '/':
			if S.ch == '/' || S.ch == '*' {
				tok, val = COMMENT, S.scanComment();
				if !S.scan_comments {
					goto loop;
				}
			} else {
				tok = S.select2(QUO, QUO_ASSIGN);
			}
		case '%': tok = S.select2(REM, REM_ASSIGN);
		case '^': tok = S.select2(XOR, XOR_ASSIGN);
		case '<':
			if S.ch == '-' {
				S.next();
				tok = ARROW;
			} else {
				tok = S.select4(LSS, LEQ, '<', SHL, SHL_ASSIGN);
			}
		case '>': tok = S.select4(GTR, GEQ, '>', SHR, SHR_ASSIGN);
		case '=': tok = S.select2(ASSIGN, EQL);
		case '!': tok = S.select2(NOT, NEQ);
		case '&': tok = S.select3(AND, AND_ASSIGN, '&', LAND);
		case '|': tok = S.select3(OR, OR_ASSIGN, '|', LOR);
		default:
			S.error(pos, "illegal character " + charString(ch));
			tok = ILLEGAL;
		}
	}

	return pos, tok, val;
}
