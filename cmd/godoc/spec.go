// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This file contains the mechanism to "linkify" html source
// text containing EBNF sections (as found in go_spec.html).
// The result is the input source text with the EBNF sections
// modified such that identifiers are linked to the respective
// definitions.

import (
	"bytes"
	"fmt"
	"io"
	"text/scanner"
)

type ebnfParser struct {
	out     io.Writer // parser output
	src     []byte    // parser input
	scanner scanner.Scanner
	prev    int    // offset of previous token
	pos     int    // offset of current token
	tok     rune   // one token look-ahead
	lit     string // token literal
}

func (p *ebnfParser) flush() {
	p.out.Write(p.src[p.prev:p.pos])
	p.prev = p.pos
}

func (p *ebnfParser) next() {
	p.tok = p.scanner.Scan()
	p.pos = p.scanner.Position.Offset
	p.lit = p.scanner.TokenText()
}

func (p *ebnfParser) printf(format string, args ...interface{}) {
	p.flush()
	fmt.Fprintf(p.out, format, args...)
}

func (p *ebnfParser) errorExpected(msg string) {
	p.printf(`<span class="highlight">error: expected %s, found %s</span>`, msg, scanner.TokenString(p.tok))
}

func (p *ebnfParser) expect(tok rune) {
	if p.tok != tok {
		p.errorExpected(scanner.TokenString(tok))
	}
	p.next() // make progress in any case
}

func (p *ebnfParser) parseIdentifier(def bool) {
	if p.tok == scanner.Ident {
		name := p.lit
		if def {
			p.printf(`<a id="%s">%s</a>`, name, name)
		} else {
			p.printf(`<a href="#%s" class="noline">%s</a>`, name, name)
		}
		p.prev += len(name) // skip identifier when printing next time
		p.next()
	} else {
		p.expect(scanner.Ident)
	}
}

func (p *ebnfParser) parseTerm() bool {
	switch p.tok {
	case scanner.Ident:
		p.parseIdentifier(false)

	case scanner.String:
		p.next()
		const ellipsis = '…' // U+2026, the horizontal ellipsis character
		if p.tok == ellipsis {
			p.next()
			p.expect(scanner.String)
		}

	case '(':
		p.next()
		p.parseExpression()
		p.expect(')')

	case '[':
		p.next()
		p.parseExpression()
		p.expect(']')

	case '{':
		p.next()
		p.parseExpression()
		p.expect('}')

	default:
		return false // no term found
	}

	return true
}

func (p *ebnfParser) parseSequence() {
	if !p.parseTerm() {
		p.errorExpected("term")
	}
	for p.parseTerm() {
	}
}

func (p *ebnfParser) parseExpression() {
	for {
		p.parseSequence()
		if p.tok != '|' {
			break
		}
		p.next()
	}
}

func (p *ebnfParser) parseProduction() {
	p.parseIdentifier(true)
	p.expect('=')
	if p.tok != '.' {
		p.parseExpression()
	}
	p.expect('.')
}

func (p *ebnfParser) parse(out io.Writer, src []byte) {
	// initialize ebnfParser
	p.out = out
	p.src = src
	p.scanner.Init(bytes.NewBuffer(src))
	p.next() // initializes pos, tok, lit

	// process source
	for p.tok != scanner.EOF {
		p.parseProduction()
	}
	p.flush()
}

// Markers around EBNF sections
var (
	openTag  = []byte(`<pre class="ebnf">`)
	closeTag = []byte(`</pre>`)
)

func Linkify(out io.Writer, src []byte) {
	for len(src) > 0 {
		// i: beginning of EBNF text (or end of source)
		i := bytes.Index(src, openTag)
		if i < 0 {
			i = len(src) - len(openTag)
		}
		i += len(openTag)

		// j: end of EBNF text (or end of source)
		j := bytes.Index(src[i:], closeTag) // close marker
		if j < 0 {
			j = len(src) - i
		}
		j += i

		// write text before EBNF
		out.Write(src[0:i])
		// process EBNF
		var p ebnfParser
		p.parse(out, src[i:j])

		// advance
		src = src[j:]
	}
}
