// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"go/doc/comment"
	"io"
)

// ToHTML converts comment text to formatted HTML.
//
// Deprecated: ToHTML cannot identify documentation links
// in the doc comment, because they depend on knowing what
// package the text came from, which is not included in this API.
//
// Given the *[doc.Package] p where text was found,
// ToHTML(w, text, nil) can be replaced by:
//
//	w.Write(p.HTML(text))
//
// which is in turn shorthand for:
//
//	w.Write(p.Printer().HTML(p.Parser().Parse(text)))
//
// If words may be non-nil, the longer replacement is:
//
//	parser := p.Parser()
//	parser.Words = words
//	w.Write(p.Printer().HTML(parser.Parse(d)))
func ToHTML(w io.Writer, text string, words map[string]string) {
	p := new(Package).Parser()
	p.Words = words
	d := p.Parse(text)
	pr := new(comment.Printer)
	w.Write(pr.HTML(d))
}

// ToText converts comment text to formatted text.
//
// Deprecated: ToText cannot identify documentation links
// in the doc comment, because they depend on knowing what
// package the text came from, which is not included in this API.
//
// Given the *[doc.Package] p where text was found,
// ToText(w, text, "", "\t", 80) can be replaced by:
//
//	w.Write(p.Text(text))
//
// In the general case, ToText(w, text, prefix, codePrefix, width)
// can be replaced by:
//
//	d := p.Parser().Parse(text)
//	pr := p.Printer()
//	pr.TextPrefix = prefix
//	pr.TextCodePrefix = codePrefix
//	pr.TextWidth = width
//	w.Write(pr.Text(d))
//
// See the documentation for [Package.Text] and [comment.Printer.Text]
// for more details.
func ToText(w io.Writer, text string, prefix, codePrefix string, width int) {
	d := new(Package).Parser().Parse(text)
	pr := &comment.Printer{
		TextPrefix:     prefix,
		TextCodePrefix: codePrefix,
		TextWidth:      width,
	}
	w.Write(pr.Text(d))
}
