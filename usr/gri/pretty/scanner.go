// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Scanner

import (
	"utf8";
	"unicode";
	"utils";
)

const (
	ILLEGAL = iota;

	IDENT;
	INT;
	FLOAT;
	STRING;
	EOF;

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
	NEQ;
	LSS;
	LEQ;
	GTR;
	GEQ;

	ASSIGN;
	DEFINE;
	NOT;
	ELLIPSIS;

	LPAREN;
	RPAREN;
	LBRACK;
	RBRACK;
	LBRACE;
	RBRACE;

	COMMA;
	SEMICOLON;
	COLON;
	PERIOD;

	// keywords
	keywords_beg;
	BREAK;
	CASE;
	CHAN;
	CONST;
	CONTINUE;

	DEFAULT;
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

	// AST use only
	EXPRSTAT;
)


func TokenString(tok int) string {
	switch tok {
	case ILLEGAL: return "ILLEGAL";

	case IDENT: return "IDENT";
	case INT: return "INT";
	case FLOAT: return "FLOAT";
	case STRING: return "STRING";
	case EOF: return "EOF";

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
	case NEQ: return "!=";
	case LSS: return "<";
	case LEQ: return "<=";
	case GTR: return ">";
	case GEQ: return ">=";

	case ASSIGN: return "=";
	case DEFINE: return ":=";
	case NOT: return "!";
	case ELLIPSIS: return "...";

	case LPAREN: return "(";
	case RPAREN: return ")";
	case LBRACK: return "[";
	case RBRACK: return "]";
	case LBRACE: return "LBRACE";
	case RBRACE: return "RBRACE";

	case COMMA: return ",";
	case SEMICOLON: return ";";
	case COLON: return ":";
	case PERIOD: return ".";

	case BREAK: return "break";
	case CASE: return "case";
	case CHAN: return "chan";
	case CONST: return "const";
	case CONTINUE: return "continue";

	case DEFAULT: return "default";
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

	case EXPRSTAT: return "EXPRSTAT";
	}

	return "token(" + Utils.IntToString(tok, 10) + ")";
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
	Warning(pos int, msg string);
}


type Scanner struct {
	// setup
	err ErrorHandler;
	src string;  // source
	scan_comments bool;

	// scanning
	pos int;  // current reading position
	ch int;  // one char look-ahead
	chpos int;  // position of ch
	linepos int;  // position of beginning of line

	// testmode
	testmode bool;
	testpos int;
}


// Read the next Unicode char into S.ch.
// S.ch < 0 means end-of-file.
func (S *Scanner) next() {
	if S.pos < len(S.src) {
		// assume ascii
		r, w := int(S.src[S.pos]), 1;
		if r >= 0x80 {
			// not ascii
			r, w = utf8.DecodeRuneInString(S.src, S.pos);
		}
		S.ch = r;
		S.chpos = S.pos;
		S.pos += w;
	} else {
		S.ch = -1;  // eof
		S.chpos = len(S.src);
	}
}


func (S *Scanner) Error(pos int, msg string) {
	// check for expected errors (test mode)
	if S.testpos < 0 || pos == S.testpos {
		// test mode:
		// S.testpos < 0:  // follow-up errors are expected and ignored
		// S.testpos == 0:  // an error is expected at S.testpos and ignored
		S.testpos = -1;
		return;
	}

	S.err.Error(pos, msg);
}


func (S *Scanner) expectNoErrors() {
	// set the next expected error position to one after eof
	// (the eof position is a legal error position!)
	S.testpos = len(S.src) + 1;
}


func (S *Scanner) Init(err ErrorHandler, src string, scan_comments, testmode bool) {
	S.err = err;
	S.src = src;
	S.scan_comments = scan_comments;

	S.pos = 0;
	S.linepos = 0;

	S.testmode = testmode;
	S.expectNoErrors();  // S.src must be set
	S.next();  // S.expectNoErrrors() must be called before
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
	return "'" + s + "' (U+" + Utils.IntToString(ch, 16) + ")";
}


func (S *Scanner) expect(ch int) {
	if S.ch != ch {
		S.Error(S.chpos, "expected " + charString(ch) + ", found " + charString(S.ch));
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


func (S *Scanner) scanComment() string {
	// first '/' already consumed
	pos := S.chpos - 1;

	if S.ch == '/' {
		//-style comment
		S.next();
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

	S.Error(pos, "comment not terminated");

exit:
	comment := S.src[pos : S.chpos];

	if S.testmode {
		// interpret ERROR and SYNC comments
		oldpos := -1;
		switch {
		case len(comment) >= 8 && comment[3 : 8] == "ERROR" :
			// an error is expected at the next token position
			oldpos = S.testpos;
			S.skipWhitespace();
			S.testpos = S.chpos;
		case len(comment) >= 7 && comment[3 : 7] == "SYNC" :
			// scanning/parsing synchronized again - no (follow-up) errors expected
			oldpos = S.testpos;
			S.expectNoErrors();
		}

		if 0 <= oldpos && oldpos <= len(S.src) {
			// the previous error was not found
			S.Error(oldpos, "ERROR not found");  // TODO this should call ErrorMsg
		}
	}

	return comment;
}


func (S *Scanner) scanIdentifier() (tok int, val string) {
	pos := S.chpos;
	for is_letter(S.ch) || digit_val(S.ch) < 10 {
		S.next();
	}
	val = S.src[pos : S.chpos];

	var present bool;
	tok, present = keywords[val];
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


func (S *Scanner) scanNumber(seen_decimal_point bool) (tok int, val string) {
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
		S.Error(S.chpos, "illegal char escape");
	}
}


func (S *Scanner) scanEscape(quote int) string {
	// TODO: fix this routine

	ch := S.ch;
	pos := S.chpos;
	S.next();
	switch ch {
	case 'a', 'b', 'f', 'n', 'r', 't', 'v', '\\':
		return string(ch);

	case '0', '1', '2', '3', '4', '5', '6', '7':
		S.scanDigits(3 - 1, 8);  // 1 char already read
		return "";  // TODO fix this

	case 'x':
		S.scanDigits(2, 16);
		return "";  // TODO fix this

	case 'u':
		S.scanDigits(4, 16);
		return "";  // TODO fix this

	case 'U':
		S.scanDigits(8, 16);
		return "";  // TODO fix this

	default:
		// check for quote outside the switch for better generated code (eventually)
		if ch == quote {
			return string(quote);
		}
		S.Error(pos, "illegal char escape");
	}

	return "";  // TODO fix this
}


func (S *Scanner) scanChar() string {
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


func (S *Scanner) scanString() string {
	// '"' already consumed

	pos := S.chpos - 1;
	for S.ch != '"' {
		ch := S.ch;
		S.next();
		if ch == '\n' || ch < 0 {
			S.Error(pos, "string not terminated");
			break;
		}
		if ch == '\\' {
			S.scanEscape('"');
		}
	}

	S.next();
	return S.src[pos : S.chpos];
}


func (S *Scanner) scanRawString() string {
	// '`' already consumed

	pos := S.chpos - 1;
	for S.ch != '`' {
		ch := S.ch;
		S.next();
		if ch == '\n' || ch < 0 {
			S.Error(pos, "string not terminated");
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


func (S *Scanner) Scan() (pos, tok int, val string) {
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
		case '\n': tok, val = COMMENT, "\n";
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
			S.Error(pos, "illegal character " + charString(ch));
			tok = ILLEGAL;
		}
	}

	return pos, tok, val;
}


type Token struct {
	Pos int;
	Tok int;
	Val string;
}


func (S *Scanner) TokenStream() <-chan *Token {
	ch := make(chan *Token, 100);
	go func(S *Scanner, ch chan <- *Token) {
		for {
			t := new(Token);
			t.Pos, t.Tok, t.Val = S.Scan();
			ch <- t;
			if t.Tok == EOF {
				break;
			}
		}
	}(S, ch);
	return ch;
}
