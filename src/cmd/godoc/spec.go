// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the mechanism to "linkify" html source
// text containing EBNF sections (as found in go_spec.html).
// The result is the input source text with the EBNF sections
// modified such that identifiers are linked to the respective
// definitions.

package main

import (
	"bytes"
	"fmt"
	"go/scanner"
	"go/token"
	"io"
)

type ebnfParser struct {
	out     io.Writer   // parser output
	src     []byte      // parser source
	file    *token.File // for position information
	scanner scanner.Scanner
	prev    int         // offset of previous token
	pos     token.Pos   // token position
	tok     token.Token // one token look-ahead
	lit     string      // token literal
}

func (p *ebnfParser) flush() {
	offs := p.file.Offset(p.pos)
	p.out.Write(p.src[p.prev:offs])
	p.prev = offs
}

func (p *ebnfParser) next() {
	if p.pos.IsValid() {
		p.flush()
	}
	p.pos, p.tok, p.lit = p.scanner.Scan()
	if p.tok.IsKeyword() {
		// TODO Should keyword mapping always happen outside scanner?
		//      Or should there be a flag to scanner to enable keyword mapping?
		p.tok = token.IDENT
	}
}

func (p *ebnfParser) Error(pos token.Position, msg string) {
	fmt.Fprintf(p.out, `<span class="alert">error: %s</span>`, msg)
}

func (p *ebnfParser) errorExpected(pos token.Pos, msg string) {
	msg = "expected " + msg
	if pos == p.pos {
		// the error happened at the current position;
		// make the error message more specific
		msg += ", found '" + p.tok.String() + "'"
		if p.tok.IsLiteral() {
			msg += " " + p.lit
		}
	}
	p.Error(p.file.Position(pos), msg)
}

func (p *ebnfParser) expect(tok token.Token) token.Pos {
	pos := p.pos
	if p.tok != tok {
		p.errorExpected(pos, "'"+tok.String()+"'")
	}
	p.next() // make progress in any case
	return pos
}

func (p *ebnfParser) parseIdentifier(def bool) {
	name := p.lit
	p.expect(token.IDENT)
	if def {
		fmt.Fprintf(p.out, `<a id="%s">%s</a>`, name, name)
	} else {
		fmt.Fprintf(p.out, `<a href="#%s" class="noline">%s</a>`, name, name)
	}
	p.prev += len(name) // skip identifier when calling flush
}

func (p *ebnfParser) parseTerm() bool {
	switch p.tok {
	case token.IDENT:
		p.parseIdentifier(false)

	case token.STRING:
		p.next()
		const ellipsis = "…" // U+2026, the horizontal ellipsis character
		if p.tok == token.ILLEGAL && p.lit == ellipsis {
			p.next()
			p.expect(token.STRING)
		}

	case token.LPAREN:
		p.next()
		p.parseExpression()
		p.expect(token.RPAREN)

	case token.LBRACK:
		p.next()
		p.parseExpression()
		p.expect(token.RBRACK)

	case token.LBRACE:
		p.next()
		p.parseExpression()
		p.expect(token.RBRACE)

	default:
		return false
	}

	return true
}

func (p *ebnfParser) parseSequence() {
	if !p.parseTerm() {
		p.errorExpected(p.pos, "term")
	}
	for p.parseTerm() {
	}
}

func (p *ebnfParser) parseExpression() {
	for {
		p.parseSequence()
		if p.tok != token.OR {
			break
		}
		p.next()
	}
}

func (p *ebnfParser) parseProduction() {
	p.parseIdentifier(true)
	p.expect(token.ASSIGN)
	if p.tok != token.PERIOD {
		p.parseExpression()
	}
	p.expect(token.PERIOD)
}

func (p *ebnfParser) parse(fset *token.FileSet, out io.Writer, src []byte) {
	// initialize ebnfParser
	p.out = out
	p.src = src
	p.file = fset.AddFile("", fset.Base(), len(src))
	p.scanner.Init(p.file, src, p, scanner.AllowIllegalChars)
	p.next() // initializes pos, tok, lit

	// process source
	for p.tok != token.EOF {
		p.parseProduction()
	}
	p.flush()
}

// Markers around EBNF sections
var (
	openTag  = []byte(`<pre class="ebnf">`)
	closeTag = []byte(`</pre>`)
)

func linkify(out io.Writer, src []byte) {
	fset := token.NewFileSet()
	for len(src) > 0 {
		n := len(src)

		// i: beginning of EBNF text (or end of source)
		i := bytes.Index(src, openTag)
		if i < 0 {
			i = n - len(openTag)
		}
		i += len(openTag)

		// j: end of EBNF text (or end of source)
		j := bytes.Index(src[i:n], closeTag) // close marker
		if j < 0 {
			j = n - i
		}
		j += i

		// write text before EBNF
		out.Write(src[0:i])
		// parse and write EBNF
		var p ebnfParser
		p.parse(fset, out, src[i:j])

		// advance
		src = src[j:n]
	}
}
