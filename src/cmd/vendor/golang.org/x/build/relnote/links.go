// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package relnote

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/mod/module"
	md "rsc.io/markdown"
)

// addSymbolLinks looks for text like [Buffer] and
// [math.Max] and replaces them with links to standard library
// symbols and packages.
// It uses the given default package for links without a package.
func addSymbolLinks(doc *md.Document, defaultPackage string) {
	addSymbolLinksBlocks(doc.Blocks, defaultPackage)
}

func addSymbolLinksBlocks(bs []md.Block, defaultPackage string) {
	for _, b := range bs {
		addSymbolLinksBlock(b, defaultPackage)
	}
}

func addSymbolLinksBlock(b md.Block, defaultPackage string) {
	switch b := b.(type) {
	case *md.Heading:
		addSymbolLinksBlock(b.Text, defaultPackage)
	case *md.Text:
		b.Inline = addSymbolLinksInlines(b.Inline, defaultPackage)
	case *md.List:
		addSymbolLinksBlocks(b.Items, defaultPackage)
	case *md.Item:
		addSymbolLinksBlocks(b.Blocks, defaultPackage)
	case *md.Paragraph:
		addSymbolLinksBlock(b.Text, defaultPackage)
	case *md.Quote:
		addSymbolLinksBlocks(b.Blocks, defaultPackage)
	// no links in these blocks
	case *md.CodeBlock:
	case *md.HTMLBlock:
	case *md.Empty:
	case *md.ThematicBreak:
	default:
		panic(fmt.Sprintf("unknown block type %T", b))
	}
}

// addSymbolLinksInlines looks for symbol links in the slice of inline markdown
// elements. It returns a new slice of inline elements with links added.
func addSymbolLinksInlines(ins []md.Inline, defaultPackage string) []md.Inline {
	var res []md.Inline
	for _, in := range ins {
		switch in := in.(type) {
		case *md.Plain:
			res = append(res, addSymbolLinksText(in.Text, defaultPackage)...)
		case *md.Strong:
			res = append(res, addSymbolLinksInlines(in.Inner, defaultPackage)...)
		case *md.Emph:
			res = append(res, addSymbolLinksInlines(in.Inner, defaultPackage)...)
		case *md.Del:
			res = append(res, addSymbolLinksInlines(in.Inner, defaultPackage)...)
		// Don't look for links in anything else.
		default:
			res = append(res, in)
		}
	}
	return res
}

// addSymbolLinksText converts symbol links in the text to markdown links.
// The text comes from a single Plain inline element, which may be split
// into multiple alternating Plain and Link elements.
func addSymbolLinksText(text, defaultPackage string) []md.Inline {
	var res []md.Inline
	last := 0

	appendPlain := func(j int) {
		if j-last > 0 {
			res = append(res, &md.Plain{Text: text[last:j]})
		}
	}

	start := -1
	for i := 0; i < len(text); i++ {
		switch text[i] {
		case '[':
			start = i
		case ']':
			link, ok := symbolLink(text[start+1:i], text[:start], text[i+1:], defaultPackage)
			if ok {
				appendPlain(start)
				res = append(res, link)
				last = i + 1
			}
			start = -1
		}

	}
	appendPlain(len(text))
	return res
}

// symbolLink convert s into a Link and returns it and true, or nil and false if
// s is not a valid link or is surrounded by runes that disqualify it from being
// converted to a link.
func symbolLink(s, before, after, defaultPackage string) (md.Inline, bool) {
	if before != "" {
		r, _ := utf8.DecodeLastRuneInString(before)
		if !isLinkAdjacentRune(r) {
			return nil, false
		}
	}
	if after != "" {
		r, _ := utf8.DecodeRuneInString(after)
		if !isLinkAdjacentRune(r) {
			return nil, false
		}
	}
	pkg, sym, ok := splitRef(s)
	if !ok {
		return nil, false
	}
	if pkg == "" {
		if defaultPackage == "" {
			return nil, false
		}
		pkg = defaultPackage
	}
	if sym != "" {
		sym = "#" + sym
	}
	return &md.Link{
		Inner: []md.Inline{&md.Plain{Text: s}},
		URL:   fmt.Sprintf("/pkg/%s%s", pkg, sym),
	}, true
}

// isLinkAdjacentRune reports whether r can be adjacent to a symbol link.
// The logic is the same as the go/doc/comment package.
func isLinkAdjacentRune(r rune) bool {
	return unicode.IsPunct(r) || r == ' ' || r == '\t' || r == '\n'
}

// splitRef splits s into a package and possibly a symbol.
// Examples:
//
//	splitRef("math.Max") => ("math", "Max", true)
//	splitRef("bytes.Buffer.String") => ("bytes", "Buffer.String", true)
//	splitRef("math") => ("math", "", true)
func splitRef(s string) (pkg, name string, ok bool) {
	s = strings.TrimPrefix(s, "*")
	pkg, name, ok = splitDocName(s)
	var recv string
	if ok {
		pkg, recv, _ = splitDocName(pkg)
	}
	if pkg != "" {
		if err := module.CheckImportPath(pkg); err != nil {
			return "", "", false
		}
	}
	if recv != "" {
		name = recv + "." + name
	}
	return pkg, name, true
}

// The following functions were copied from go/doc/comment/parse.go.

// If text is of the form before.Name, where Name is a capitalized Go identifier,
// then splitDocName returns before, name, true.
// Otherwise it returns text, "", false.
func splitDocName(text string) (before, name string, foundDot bool) {
	i := strings.LastIndex(text, ".")
	name = text[i+1:]
	if !isName(name) {
		return text, "", false
	}
	if i >= 0 {
		before = text[:i]
	}
	return before, name, true
}

// isName reports whether s is a capitalized Go identifier (like Name).
func isName(s string) bool {
	t, ok := ident(s)
	if !ok || t != s {
		return false
	}
	r, _ := utf8.DecodeRuneInString(s)
	return unicode.IsUpper(r)
}

// ident checks whether s begins with a Go identifier.
// If so, it returns the identifier, which is a prefix of s, and ok == true.
// Otherwise it returns "", false.
// The caller should skip over the first len(id) bytes of s
// before further processing.
func ident(s string) (id string, ok bool) {
	// Scan [\pL_][\pL_0-9]*
	n := 0
	for n < len(s) {
		if c := s[n]; c < utf8.RuneSelf {
			if isIdentASCII(c) && (n > 0 || c < '0' || c > '9') {
				n++
				continue
			}
			break
		}
		r, nr := utf8.DecodeRuneInString(s[n:])
		if unicode.IsLetter(r) {
			n += nr
			continue
		}
		break
	}
	return s[:n], n > 0
}

// isIdentASCII reports whether c is an ASCII identifier byte.
func isIdentASCII(c byte) bool {
	// mask is a 128-bit bitmap with 1s for allowed bytes,
	// so that the byte c can be tested with a shift and an and.
	// If c > 128, then 1<<c and 1<<(c-64) will both be zero,
	// and this function will return false.
	const mask = 0 |
		(1<<26-1)<<'A' |
		(1<<26-1)<<'a' |
		(1<<10-1)<<'0' |
		1<<'_'

	return ((uint64(1)<<c)&(mask&(1<<64-1)) |
		(uint64(1)<<(c-64))&(mask>>64)) != 0
}
