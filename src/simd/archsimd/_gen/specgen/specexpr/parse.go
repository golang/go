// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specexpr

import (
	"fmt"
	"strconv"
	"strings"
)

// ParseExpr parses a constraint expression.
func ParseExpr(x string) (Expr, error) {
	var e Expr
	var err error

	func() {
		defer func() {
			v := recover()
			if v2, ok := v.(*parseError); ok {
				err = v2
				return
			} else if v != nil {
				panic(v)
			}
		}()
		p := &parser{s: x}
		p.skipSpace()
		e = p.parseExpr()
		if p.pos < len(p.s) {
			p.fail("unexpected trailing characters")
		}
	}()
	return e, err
}

type parser struct {
	s   string
	pos int
}

type parseError struct {
	msg  string
	args []any
	pos  int
}

func (e *parseError) Error() string {
	return fmt.Sprintf("%s at %d", fmt.Sprintf(e.msg, e.args...), 1+e.pos)
}

func (p *parser) fail(msg string, args ...any) {
	panic(&parseError{msg: msg, args: args, pos: p.pos})
}

func (p *parser) failAt(pos int, msg string, args ...any) {
	panic(&parseError{msg: msg, args: args, pos: pos})
}

func (p *parser) skipSpace() {
	for p.pos < len(p.s) && (p.s[p.pos] == ' ' || p.s[p.pos] == '\t' || p.s[p.pos] == '\r' || p.s[p.pos] == '\n') {
		p.pos++
	}
}

func (p *parser) peek() byte {
	if p.pos >= len(p.s) {
		return 0
	}
	return p.s[p.pos]
}

// consume consumes one or more characters matching pred. It does NOT consume
// whitespace.
func (p *parser) consume(pred func(c byte) bool) (string, bool) {
	if p.pos >= len(p.s) || !pred(p.s[p.pos]) {
		return "", false
	}
	start := p.pos
	p.pos++
	for p.pos < len(p.s) && pred(p.s[p.pos]) {
		p.pos++
	}
	return p.s[start:p.pos], true
}

func isDigit(c byte) bool {
	return c >= '0' && c <= '9'
}

func isAlpha(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

// try either consumes s followed zero or more whitespace and returns true, or
// does nothing and returns false.
func (p *parser) try(s string) bool {
	if !strings.HasPrefix(p.s[p.pos:], s) {
		return false
	}
	p.pos += len(s)
	p.skipSpace()
	return true
}

func (p *parser) peekShape() bool {
	// Peek base type
	if p.peek() == '{' {
		return true
	}
	start := p.pos
	if _, ok := p.consume(isAlpha); !ok {
		return false
	}
	// Peek N
	ok := p.pos < len(p.s) && (isDigit(p.s[p.pos]) || p.s[p.pos] == '{')
	p.pos = start
	return ok
}

func (p *parser) parseExpr() Expr {
	return p.parseComparison()
}

func (p *parser) parseComparison() Expr {
	x := p.parseMulDiv()
	var op BinOp
	switch {
	case p.try("="):
		op = OpEqual
	case p.try("!="):
		op = OpNotEqual
	case p.try(">="):
		op = OpGreaterOrEqual
	case p.try(">"):
		op = OpGreaterThan
	case p.try("<="):
		op = OpLessOrEqual
	case p.try("<"):
		op = OpLessThan
	default:
		return x
	}

	y := p.parseMulDiv()
	return &BinExpr{op, x, y}
}

func (p *parser) parseMulDiv() Expr {
	x := p.parsePrimary()
loop:
	for {
		var op BinOp
		switch {
		case p.try("*"):
			op = OpTimes
		case p.try("/"):
			op = OpDiv
		default:
			break loop
		}
		y := p.parsePrimary()
		x = &BinExpr{op, x, y}
	}
	return x
}

func (p *parser) parsePrimary() Expr {
	if p.try("(") {
		e := p.parseExpr()
		if !p.try(")") {
			p.fail("expected ')'")
		}
		return e
	}

	// Shape
	if p.peekShape() {
		return p.parseSymShape()
	}

	// Number literal
	b := p.peek()
	if isDigit(b) {
		return p.parseNumber()
	}

	// Variable
	if isAlpha(b) {
		name, _ := p.consume(isAlpha)
		p.skipSpace()
		return Variable(name)
	}

	if b == 0 {
		p.fail("unexpected end")
	}
	p.fail("unexpected character '%c'", b)
	panic("not reachable")
}

func (p *parser) parseNumber() Int {
	nStr, ok := p.consume(isDigit)
	if !ok {
		p.fail("expected number")
	}
	num, err := strconv.Atoi(nStr)
	if err != nil {
		p.fail("%s", err)
	}
	p.skipSpace()
	return Int(num)
}

// - BaseNxL: A fixed vector with L lanes. E.g., Int32x4
// - BaseNs: A scalable vector. E.g., Float32s
// - BaseNwW: A fixed vector of width W. E.g., Int32w128 (same as Int32x4)
// - MaskNxL, MaskNs, or MaskNwW: Similar, but describes a mask.
// - baseN: A scalar. E.g., uint8
func (p *parser) parseSymShape() *Apply {
	var b, n Expr

	trySymPart := func() Expr {
		openPos := p.pos
		if !p.try("{") {
			return nil
		}

		x := p.parseExpr()
		if !p.try("}") {
			p.failAt(openPos, "'{' missing close '}' in symbolic shape")
		}
		return x
	}

	// Base
	if b = trySymPart(); b == nil {
		base, ok := p.consume(isAlpha)
		if !ok {
			p.fail("expected shape base name matching [a-zA-Z]+")
		}
		b = &Literal{strings.ToLower(base)}
	}

	// Element size
	if n = trySymPart(); n == nil {
		n = p.parseNumber()
	}

	elem := MakeBasic(b, n)

	// Width
	var x *Apply
	// Don't use p.try here because that will skip whitespace.
	switch p.peek() {
	case 's':
		p.pos++
		x = MakeVector(elem, VW())
	case 'x':
		p.pos++
		l := trySymPart()
		if l == nil {
			// TODO: Disallow width-rounding in this case?
			l = p.parseNumber()
		}
		x = makeVectorL(elem, l)
	case 'w':
		p.pos++
		w := trySymPart()
		if w == nil {
			w = p.parseNumber()
		}
		x = MakeVector(elem, w)
	default:
		// Scalar
		x = elem
	}

	p.skipSpace()
	return x
}
