// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strings"
)

// exprParser is a //go:build expression parser and evaluator.
// The parser is a trivial precedence-based parser which is still
// almost overkill for these very simple expressions.
type exprParser struct {
	x string
	t exprToken // upcoming token
}

// val is the value type result of parsing.
// We don't keep a parse tree, just the value of the expression.
type val bool

// exprToken describes a single token in the input.
// Prefix operators define a prefix func that parses the
// upcoming value. Binary operators define an infix func
// that combines two values according to the operator.
// In that case, the parsing loop parses the two values.
type exprToken struct {
	tok    string
	prec   int
	prefix func(*exprParser) val
	infix  func(val, val) val
}

var exprTokens []exprToken

func init() { // init to break init cycle
	exprTokens = []exprToken{
		{tok: "&&", prec: 1, infix: func(x, y val) val { return x && y }},
		{tok: "||", prec: 2, infix: func(x, y val) val { return x || y }},
		{tok: "!", prec: 3, prefix: (*exprParser).not},
		{tok: "(", prec: 3, prefix: (*exprParser).paren},
		{tok: ")"},
	}
}

// matchexpr parses and evaluates the //go:build expression x.
func matchexpr(x string) (matched bool, err error) {
	defer func() {
		if e := recover(); e != nil {
			matched = false
			err = fmt.Errorf("parsing //go:build line: %v", e)
		}
	}()

	p := &exprParser{x: x}
	p.next()
	v := p.parse(0)
	if p.t.tok != "end of expression" {
		panic("unexpected " + p.t.tok)
	}
	return bool(v), nil
}

// parse parses an expression, including binary operators at precedence >= prec.
func (p *exprParser) parse(prec int) val {
	if p.t.prefix == nil {
		panic("unexpected " + p.t.tok)
	}
	v := p.t.prefix(p)
	for p.t.prec >= prec && p.t.infix != nil {
		t := p.t
		p.next()
		v = t.infix(v, p.parse(t.prec+1))
	}
	return v
}

// not is the prefix parser for a ! token.
func (p *exprParser) not() val {
	p.next()
	return !p.parse(100)
}

// paren is the prefix parser for a ( token.
func (p *exprParser) paren() val {
	p.next()
	v := p.parse(0)
	if p.t.tok != ")" {
		panic("missing )")
	}
	p.next()
	return v
}

// next advances the parser to the next token,
// leaving the token in p.t.
func (p *exprParser) next() {
	p.x = strings.TrimSpace(p.x)
	if p.x == "" {
		p.t = exprToken{tok: "end of expression"}
		return
	}
	for _, t := range exprTokens {
		if strings.HasPrefix(p.x, t.tok) {
			p.x = p.x[len(t.tok):]
			p.t = t
			return
		}
	}

	i := 0
	for i < len(p.x) && validtag(p.x[i]) {
		i++
	}
	if i == 0 {
		panic(fmt.Sprintf("syntax error near %#q", rune(p.x[i])))
	}
	tag := p.x[:i]
	p.x = p.x[i:]
	p.t = exprToken{
		tok: "tag",
		prefix: func(p *exprParser) val {
			p.next()
			return val(matchtag(tag))
		},
	}
}

func validtag(c byte) bool {
	return 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || '0' <= c && c <= '9' || c == '.' || c == '_'
}
