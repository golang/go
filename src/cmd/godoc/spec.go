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
	"bytes";
	"fmt";
	"go/scanner";
	"go/token";
	"io";
	"strings";
)


type ebnfParser struct {
	out	io.Writer;	// parser output
	src	[]byte;		// parser source
	scanner	scanner.Scanner;
	prev	int;		// offset of previous token
	pos	token.Position;	// token position
	tok	token.Token;	// one token look-ahead
	lit	[]byte;		// token literal
}


func (p *ebnfParser) flush() {
	p.out.Write(p.src[p.prev:p.pos.Offset]);
	p.prev = p.pos.Offset;
}


func (p *ebnfParser) next() {
	p.flush();
	p.pos, p.tok, p.lit = p.scanner.Scan();
	if p.tok.IsKeyword() {
		// TODO Should keyword mapping always happen outside scanner?
		//      Or should there be a flag to scanner to enable keyword mapping?
		p.tok = token.IDENT
	}
}


func (p *ebnfParser) Error(pos token.Position, msg string) {
	fmt.Fprintf(p.out, `<span class="alert">error: %s</span>`, msg)
}


func (p *ebnfParser) errorExpected(pos token.Position, msg string) {
	msg = "expected " + msg;
	if pos.Offset == p.pos.Offset {
		// the error happened at the current position;
		// make the error message more specific
		msg += ", found '" + p.tok.String() + "'";
		if p.tok.IsLiteral() {
			msg += " " + string(p.lit)
		}
	}
	p.Error(pos, msg);
}


func (p *ebnfParser) expect(tok token.Token) token.Position {
	pos := p.pos;
	if p.tok != tok {
		p.errorExpected(pos, "'"+tok.String()+"'")
	}
	p.next();	// make progress in any case
	return pos;
}


func (p *ebnfParser) parseIdentifier(def bool) {
	name := string(p.lit);
	p.expect(token.IDENT);
	if def {
		fmt.Fprintf(p.out, `<a id="%s">%s</a>`, name, name)
	} else {
		fmt.Fprintf(p.out, `<a href="#%s" class="noline">%s</a>`, name, name)
	}
	p.prev += len(name);	// skip identifier when calling flush
}


func (p *ebnfParser) parseTerm() bool {
	switch p.tok {
	case token.IDENT:
		p.parseIdentifier(false)

	case token.STRING:
		p.next();
		if p.tok == token.ELLIPSIS {
			p.next();
			p.expect(token.STRING);
		}

	case token.LPAREN:
		p.next();
		p.parseExpression();
		p.expect(token.RPAREN);

	case token.LBRACK:
		p.next();
		p.parseExpression();
		p.expect(token.RBRACK);

	case token.LBRACE:
		p.next();
		p.parseExpression();
		p.expect(token.RBRACE);

	default:
		return false
	}

	return true;
}


func (p *ebnfParser) parseSequence() {
	for p.parseTerm() {
	}
}


func (p *ebnfParser) parseExpression() {
	for {
		p.parseSequence();
		if p.tok != token.OR {
			break
		}
		p.next();
	}
}


func (p *ebnfParser) parseProduction() {
	p.parseIdentifier(true);
	p.expect(token.ASSIGN);
	p.parseExpression();
	p.expect(token.PERIOD);
}


func (p *ebnfParser) parse(out io.Writer, src []byte) {
	// initialize ebnfParser
	p.out = out;
	p.src = src;
	p.scanner.Init("", src, p, 0);
	p.next();	// initializes pos, tok, lit

	// process source
	for p.tok != token.EOF {
		p.parseProduction()
	}
	p.flush();
}


// Markers around EBNF sections
var (
	openTag		= strings.Bytes(`<pre class="ebnf">`);
	closeTag	= strings.Bytes(`</pre>`);
)


func linkify(out io.Writer, src []byte) {
	for len(src) > 0 {
		n := len(src);

		// i: beginning of EBNF text (or end of source)
		i := bytes.Index(src, openTag);
		if i < 0 {
			i = n - len(openTag)
		}
		i += len(openTag);

		// j: end of EBNF text (or end of source)
		j := bytes.Index(src[i:n], closeTag);	// close marker
		if j < 0 {
			j = n - i
		}
		j += i;

		// write text before EBNF
		out.Write(src[0:i]);
		// parse and write EBNF
		var p ebnfParser;
		p.parse(out, src[i:j]);

		// advance
		src = src[j:n];
	}
}
