// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "unicode/utf8"

// This file holds the loGo/s parser logic for M7-S1 (`const if`) and
// M7-S4 (`const switch T`; SPEC Sec 2.5, T-205). parser.go calls in
// through one hook line in the statement `_Const` case (G3); every rule
// below runs solely in LogosMode, so plain Go parses exactly as
// upstream. The sibling M7-S2 slice owns `const for` through its own
// hook line and reuses logosPeekConstNext, which never consumes input,
// so hook order is irrelevant and either hook falls through to the
// upstream const declaration path on a non-match.

// constPeek is a read-only cursor over the source bytes that follow the
// current token. It fills the scanner buffer as it needs bytes and keeps
// both its read offset and its word mark valid across the compaction of
// each fill, so the scanner state stays untouched.
type constPeek struct {
	s    *scanner
	r    int
	mark int
}

// logosPeekConstNext reports the keyword that follows the current
// `const` token without consuming any input, so independent `const`
// statement hooks can claim their keyword and fall through otherwise.
// It skips blanks and comments with a read-only byte scan in the
// logosLeadingDot fillPeek style, starting at the character the scanner
// has already read, and reads one ASCII word: an exact `if`, `switch`,
// `for`, or `func` yields _If, _Switch, _For, or _Func; anything else
// (including a longer identifier such as `inform`, where the word
// boundary fails, or EOF) yields _EOF, which no caller claims.
func logosPeekConstNext(p *parser) token {
	k := constPeek{s: &p.scanner}
	k.r = k.s.r - k.s.chw
	if !k.skipBlank() {
		return _EOF
	}
	return logosConstKeyword(k.word())
}

// logosConstKeyword maps the word after `const` to its keyword token, or
// to _EOF when no hook claims it.
func logosConstKeyword(word string) token {
	switch word {
	case "if":
		return _If
	case "switch":
		return _Switch
	case "for":
		return _For
	case "func":
		return _Func
	}
	return _EOF
}

// avail makes the byte at the cursor readable, filling the buffer when
// the cursor stands at its end. It reports false at the end of input.
func (k *constPeek) avail() bool {
	for k.r >= k.s.e {
		if k.s.ioerr != nil {
			return false
		}
		shift := k.s.fillPeek()
		k.r -= shift
		k.mark -= shift
	}
	return true
}

// skipBlank moves the cursor over blanks and comments to the next byte
// that starts something else. It reports false at the end of input and
// at a `/` that opens no comment.
func (k *constPeek) skipBlank() bool {
	for k.avail() {
		switch c := k.s.buf[k.r]; c {
		case ' ', '\t', '\n', '\r':
			k.r++
		case '/':
			if !k.skipComment() {
				return false
			}
		default:
			return true
		}
	}
	return false
}

// skipComment moves the cursor past the comment that starts at its `/`.
// It reports false at the end of input and when the `/` opens no comment.
func (k *constPeek) skipComment() bool {
	k.r++
	if !k.avail() {
		return false
	}
	switch k.s.buf[k.r] {
	case '/':
		return k.skipLine()
	case '*':
		return k.skipBlock()
	}
	return false
}

// skipLine moves the cursor to the newline that ends a line comment. It
// reports false at the end of input.
func (k *constPeek) skipLine() bool {
	for k.avail() {
		if k.s.buf[k.r] == '\n' {
			return true
		}
		k.r++
	}
	return false
}

// skipBlock moves the cursor past the `*/` that ends a block comment whose
// `/*` the cursor stands in. It reports false at the end of input.
func (k *constPeek) skipBlock() bool {
	k.r++
	for k.avail() {
		if k.s.buf[k.r] != '*' {
			k.r++
			continue
		}
		k.r++
		if !k.avail() {
			return false
		}
		if k.s.buf[k.r] == '/' {
			k.r++
			return true
		}
	}
	return false
}

// word reads the ASCII word at the cursor. It returns the empty string
// when the word is followed by a non-ASCII byte, because that byte could
// continue an identifier and so the word is no keyword.
func (k *constPeek) word() string {
	k.mark = k.r
	for k.avail() && isWordByte(k.s.buf[k.r]) {
		k.r++
	}
	if k.avail() && k.s.buf[k.r] >= utf8.RuneSelf {
		return ""
	}
	return string(k.s.buf[k.mark:k.r])
}

// isWordByte reports whether c is an ASCII identifier byte.
func isWordByte(c byte) bool {
	return c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9'
}

// logosConstStmt recognizes a `const if` (M7-S1) or `const switch`
// (M7-S4) statement and returns it. It returns nil when the current
// token is not `const` in LogosMode, when the peeked keyword is neither
// `if` nor `switch`, or outside LogosMode, consuming nothing in every
// nil case, so the caller falls through to the upstream const
// declaration path and the sibling `const for` hook stays independent.
func (p *parser) logosConstStmt() Stmt {
	if p.mode&LogosMode == 0 || p.tok != _Const {
		return nil
	}
	switch logosPeekConstNext(p) {
	case _If:
		return p.constIfStmt()
	case _Switch:
		return p.constSwitchStmt()
	}
	return nil
}

// constIfStmt parses a `const if` statement with the `const` keyword
// current:
//
//	ConstIf = "const" "if" Expr Block ["else" (Block | ConstIf)] .
//
// The condition is a full expression, parsed with composite literals
// off like an `if` header, so `const if name {` ends the condition at
// the name; types2 requires it to be a constant boolean. There is no
// initializer (unlike header). The else arm takes a block or a nested
// `const if` per the SPEC grammar; a regular `else if` stays a syntax
// error.
func (p *parser) constIfStmt() Stmt {
	if trace {
		defer p.trace("constIfStmt")()
	}

	s := new(ConstIfStmt)
	s.pos = p.pos()
	p.next()
	p.next()
	outer := p.xnest
	p.xnest = -1
	s.Cond = p.expr()
	p.xnest = outer
	s.Then = p.blockStmt("const if clause")
	if p.got(_Else) {
		s.Else = p.constElse()
	}
	return s
}

// constElse parses the arm after `else` in a `const if`: a block or a
// nested `const if`. Anything else is a syntax error, reported once.
func (p *parser) constElse() Stmt {
	switch {
	case p.tok == _Const && logosPeekConstNext(p) == _If:
		return p.constIfStmt()
	case p.tok == _Lbrace:
		return p.blockStmt("")
	}
	p.syntaxError("else must be followed by a statement block or const if")
	p.advance(_Rbrace)
	return nil
}

// constSwitchStmt parses a `const switch` statement with the `const`
// keyword current:
//
//	ConstSwitch = "const" "switch" Type "{" { "case" ["~"] Type ":" Stmts } "}" .
//
// The tag is a type (usually the name of a type parameter). Each case
// names exactly one type: a comma list is a syntax error, and so are
// `default` arms (the SPEC grammar lists cases only; an unmatched type
// falls through to the statements after the switch). A leading `~`
// marks an underlying-type case.
func (p *parser) constSwitchStmt() Stmt {
	if trace {
		defer p.trace("constSwitchStmt")()
	}

	s := new(ConstSwitchStmt)
	s.pos = p.pos()
	p.next()
	p.next()
	s.Targ = p.type_()
	s.Lbrace = p.pos()
	if !p.got(_Lbrace) {
		p.syntaxError("missing { after const switch type")
		p.advance(_Case, _Rbrace)
	}
	for p.tok != _EOF && p.tok != _Rbrace {
		s.Cases = append(s.Cases, p.constCase())
	}
	s.Rbrace = p.pos()
	p.want(_Rbrace)
	return s
}

// constCase parses one `const switch` case: `case` with an optional `~`
// and exactly one type, a colon, and the case statements.
func (p *parser) constCase() *ConstSwitchCase {
	if trace {
		defer p.trace("constCase")()
	}

	c := new(ConstSwitchCase)
	c.Case = p.pos()
	if p.tok != _Case {
		return p.constCaseRecover(c)
	}
	p.next()
	if p.tok == _Operator && p.op == Tilde {
		c.Tilde = true
		c.TildePos = p.pos()
		p.next()
	}
	c.Type = p.type_()
	if p.tok == _Comma {
		p.syntaxError("const switch case takes a single type, not a list")
		p.advance(_Colon, _Case, _Rbrace)
	}
	c.Colon = p.pos()
	p.want(_Colon)
	c.Body = p.stmtList()
	return c
}

// constCaseRecover reports a clause that does not start with `case`, such
// as a `default` arm, which the grammar does not have. It skips to the
// clause's colon and parses the statements after it, so the clause costs
// one diagnostic and not one for each token it holds.
func (p *parser) constCaseRecover(c *ConstSwitchCase) *ConstSwitchCase {
	p.syntaxError("expected case or }")
	p.advance(_Colon, _Case, _Rbrace)
	c.Type = p.badExpr()
	c.Colon = p.pos()
	if p.got(_Colon) {
		c.Body = p.stmtList()
	}
	return c
}
