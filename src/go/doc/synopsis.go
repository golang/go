// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"go/doc/comment"
	"strings"
	"unicode"
)

// firstSentence returns the first sentence in s.
// The sentence ends after the first period followed by space and
// not preceded by exactly one uppercase letter.
func firstSentence(s string) string {
	var ppp, pp, p rune
	for i, q := range s {
		if q == '\n' || q == '\r' || q == '\t' {
			q = ' '
		}
		if q == ' ' && p == '.' && (!unicode.IsUpper(pp) || unicode.IsUpper(ppp)) {
			return s[:i]
		}
		if p == '。' || p == '．' {
			return s[:i]
		}
		ppp, pp, p = pp, p, q
	}
	return s
}

// Synopsis returns a cleaned version of the first sentence in text.
//
// Deprecated: New programs should use [Package.Synopsis] instead,
// which handles links in text properly.
func Synopsis(text string) string {
	var p Package
	return p.Synopsis(text)
}

// IllegalPrefixes is a list of lower-case prefixes that identify
// a comment as not being a doc comment.
// This helps to avoid misinterpreting the common mistake
// of a copyright notice immediately before a package statement
// as being a doc comment.
var IllegalPrefixes = []string{
	"copyright",
	"all rights",
	"author",
}

// Synopsis returns a cleaned version of the first sentence in text.
// That sentence ends after the first period followed by space and not
// preceded by exactly one uppercase letter, or at the first paragraph break.
// The result string has no \n, \r, or \t characters and uses only single
// spaces between words. If text starts with any of the [IllegalPrefixes],
// the result is the empty string.
func (p *Package) Synopsis(text string) string {
	text = firstSentence(text)
	lower := strings.ToLower(text)
	for _, prefix := range IllegalPrefixes {
		if strings.HasPrefix(lower, prefix) {
			return ""
		}
	}
	pr := p.Printer()
	pr.TextWidth = -1
	d := p.Parser().Parse(text)
	if len(d.Content) == 0 {
		return ""
	}
	if _, ok := d.Content[0].(*comment.Paragraph); !ok {
		return ""
	}
	d.Content = d.Content[:1] // might be blank lines, code blocks, etc in “first sentence”
	return strings.TrimSpace(string(pr.Text(d)))
}
