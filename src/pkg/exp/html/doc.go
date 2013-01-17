// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package html implements an HTML5-compliant tokenizer and parser.

Tokenization is done by creating a Tokenizer for an io.Reader r. It is the
caller's responsibility to ensure that r provides UTF-8 encoded HTML.

	z := html.NewTokenizer(r)

Given a Tokenizer z, the HTML is tokenized by repeatedly calling z.Next(),
which parses the next token and returns its type, or an error:

	for {
		tt := z.Next()
		if tt == html.ErrorToken {
			// ...
			return ...
		}
		// Process the current token.
	}

There are two APIs for retrieving the current token. The high-level API is to
call Token; the low-level API is to call Text or TagName / TagAttr. Both APIs
allow optionally calling Raw after Next but before Token, Text, TagName, or
TagAttr. In EBNF notation, the valid call sequence per token is:

	Next {Raw} [ Token | Text | TagName {TagAttr} ]

Token returns an independent data structure that completely describes a token.
Entities (such as "&lt;") are unescaped, tag names and attribute keys are
lower-cased, and attributes are collected into a []Attribute. For example:

	for {
		if z.Next() == html.ErrorToken {
			// Returning io.EOF indicates success.
			return z.Err()
		}
		emitToken(z.Token())
	}

The low-level API performs fewer allocations and copies, but the contents of
the []byte values returned by Text, TagName and TagAttr may change on the next
call to Next. For example, to extract an HTML page's anchor text:

	depth := 0
	for {
		tt := z.Next()
		switch tt {
		case ErrorToken:
			return z.Err()
		case TextToken:
			if depth > 0 {
				// emitBytes should copy the []byte it receives,
				// if it doesn't process it immediately.
				emitBytes(z.Text())
			}
		case StartTagToken, EndTagToken:
			tn, _ := z.TagName()
			if len(tn) == 1 && tn[0] == 'a' {
				if tt == StartTagToken {
					depth++
				} else {
					depth--
				}
			}
		}
	}

Parsing is done by calling Parse with an io.Reader, which returns the root of
the parse tree (the document element) as a *Node. It is the caller's
responsibility to ensure that the Reader provides UTF-8 encoded HTML. For
example, to process each anchor node in depth-first order:

	doc, err := html.Parse(r)
	if err != nil {
		// ...
	}
	var f func(*html.Node)
	f = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "a" {
			// Do something with n...
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			f(c)
		}
	}
	f(doc)

The relevant specifications include:
http://www.whatwg.org/specs/web-apps/current-work/multipage/syntax.html and
http://www.whatwg.org/specs/web-apps/current-work/multipage/tokenization.html
*/
package html

// The tokenization algorithm implemented by this package is not a line-by-line
// transliteration of the relatively verbose state-machine in the WHATWG
// specification. A more direct approach is used instead, where the program
// counter implies the state, such as whether it is tokenizing a tag or a text
// node. Specification compliance is verified by checking expected and actual
// outputs over a test suite rather than aiming for algorithmic fidelity.

// TODO(nigeltao): Does a DOM API belong in this package or a separate one?
// TODO(nigeltao): How does parsing interact with a JavaScript engine?
