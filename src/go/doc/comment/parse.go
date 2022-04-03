// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package comment

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// A Doc is a parsed Go doc comment.
type Doc struct {
	// Content is the sequence of content blocks in the comment.
	Content []Block

	// Links is the link definitions in the comment.
	Links []*LinkDef
}

// A LinkDef is a single link definition.
type LinkDef struct {
	Text string // the link text
	URL  string // the link URL
	Used bool   // whether the comment uses the definition
}

// A Block is block-level content in a doc comment,
// one of *[Code], *[Heading], *[List], or *[Paragraph].
type Block interface {
	block()
}

// A Heading is a doc comment heading.
type Heading struct {
	Text []Text // the heading text
}

func (*Heading) block() {}

// A List is a numbered or bullet list.
// Lists are always non-empty: len(Items) > 0.
// In a numbered list, every Items[i].Number is a non-empty string.
// In a bullet list, every Items[i].Number is an empty string.
type List struct {
	// Items is the list items.
	Items []*ListItem

	// ForceBlankBefore indicates that the list must be
	// preceded by a blank line when reformatting the comment,
	// overriding the usual conditions. See the BlankBefore method.
	//
	// The comment parser sets ForceBlankBefore for any list
	// that is preceded by a blank line, to make sure
	// the blank line is preserved when printing.
	ForceBlankBefore bool

	// ForceBlankBetween indicates that list items must be
	// separated by blank lines when reformatting the comment,
	// overriding the usual conditions. See the BlankBetween method.
	//
	// The comment parser sets ForceBlankBetween for any list
	// that has a blank line between any two of its items, to make sure
	// the blank lines are preserved when printing.
	ForceBlankBetween bool
}

func (*List) block() {}

// BlankBefore reports whether a reformatting of the comment
// should include a blank line before the list.
// The default rule is the same as for [BlankBetween]:
// if the list item content contains any blank lines
// (meaning at least one item has multiple paragraphs)
// then the list itself must be preceded by a blank line.
// A preceding blank line can be forced by setting [List].ForceBlankBefore.
func (l *List) BlankBefore() bool {
	return l.ForceBlankBefore || l.BlankBetween()
}

// BlankBetween reports whether a reformatting of the comment
// should include a blank line between each pair of list items.
// The default rule is that if the list item content contains any blank lines
// (meaning at least one item has multiple paragraphs)
// then list items must themselves be separated by blank lines.
// Blank line separators can be forced by setting [List].ForceBlankBetween.
func (l *List) BlankBetween() bool {
	if l.ForceBlankBetween {
		return true
	}
	for _, item := range l.Items {
		if len(item.Content) != 1 {
			// Unreachable for parsed comments today,
			// since the only way to get multiple item.Content
			// is multiple paragraphs, which must have been
			// separated by a blank line.
			return true
		}
	}
	return false
}

// A ListItem is a single item in a numbered or bullet list.
type ListItem struct {
	// Number is a decimal string in a numbered list
	// or an empty string in a bullet list.
	Number string // "1", "2", ...; "" for bullet list

	// Content is the list content.
	// Currently, restrictions in the parser and printer
	// require every element of Content to be a *Paragraph.
	Content []Block // Content of this item.
}

// A Paragraph is a paragraph of text.
type Paragraph struct {
	Text []Text
}

func (*Paragraph) block() {}

// A Code is a preformatted code block.
type Code struct {
	// Text is the preformatted text, ending with a newline character.
	// It may be multiple lines, each of which ends with a newline character.
	// It is never empty, nor does it start or end with a blank line.
	Text string
}

func (*Code) block() {}

// A Text is text-level content in a doc comment,
// one of [Plain], [Italic], *[Link], or *[DocLink].
type Text interface {
	text()
}

// A Plain is a string rendered as plain text (not italicized).
type Plain string

func (Plain) text() {}

// An Italic is a string rendered as italicized text.
type Italic string

func (Italic) text() {}

// A Link is a link to a specific URL.
type Link struct {
	Auto bool   // is this an automatic (implicit) link of a literal URL?
	Text []Text // text of link
	URL  string // target URL of link
}

func (*Link) text() {}

// A DocLink is a link to documentation for a Go package or symbol.
type DocLink struct {
	Text []Text // text of link

	// ImportPath, Recv, and Name identify the Go package or symbol
	// that is the link target. The potential combinations of
	// non-empty fields are:
	//  - ImportPath: a link to another package
	//  - ImportPath, Name: a link to a const, func, type, or var in another package
	//  - ImportPath, Recv, Name: a link to a method in another package
	//  - Name: a link to a const, func, type, or var in this package
	//  - Recv, Name: a link to a method in this package
	ImportPath string // import path
	Recv       string // receiver type, without any pointer star, for methods
	Name       string // const, func, type, var, or method name
}

func (*DocLink) text() {}

// leadingSpace returns the longest prefix of s consisting of spaces and tabs.
func leadingSpace(s string) string {
	i := 0
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	return s[:i]
}

// isOldHeading reports whether line is an old-style section heading.
// line is all[off].
func isOldHeading(line string, all []string, off int) bool {
	if off <= 0 || all[off-1] != "" || off+2 >= len(all) || all[off+1] != "" || leadingSpace(all[off+2]) != "" {
		return false
	}

	line = strings.TrimSpace(line)

	// a heading must start with an uppercase letter
	r, _ := utf8.DecodeRuneInString(line)
	if !unicode.IsLetter(r) || !unicode.IsUpper(r) {
		return false
	}

	// it must end in a letter or digit:
	r, _ = utf8.DecodeLastRuneInString(line)
	if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
		return false
	}

	// exclude lines with illegal characters. we allow "(),"
	if strings.ContainsAny(line, ";:!?+*/=[]{}_^°&§~%#@<\">\\") {
		return false
	}

	// allow "'" for possessive "'s" only
	for b := line; ; {
		var ok bool
		if _, b, ok = strings.Cut(b, "'"); !ok {
			break
		}
		if b != "s" && !strings.HasPrefix(b, "s ") {
			return false // ' not followed by s and then end-of-word
		}
	}

	// allow "." when followed by non-space
	for b := line; ; {
		var ok bool
		if _, b, ok = strings.Cut(b, "."); !ok {
			break
		}
		if b == "" || strings.HasPrefix(b, " ") {
			return false // not followed by non-space
		}
	}

	return true
}

// autoURL checks whether s begins with a URL that should be hyperlinked.
// If so, it returns the URL, which is a prefix of s, and ok == true.
// Otherwise it returns "", false.
// The caller should skip over the first len(url) bytes of s
// before further processing.
func autoURL(s string) (url string, ok bool) {
	// Find the ://. Fast path to pick off non-URL,
	// since we call this at every position in the string.
	// The shortest possible URL is ftp://x, 7 bytes.
	var i int
	switch {
	case len(s) < 7:
		return "", false
	case s[3] == ':':
		i = 3
	case s[4] == ':':
		i = 4
	case s[5] == ':':
		i = 5
	case s[6] == ':':
		i = 6
	default:
		return "", false
	}
	if i+3 > len(s) || s[i:i+3] != "://" {
		return "", false
	}

	// Check valid scheme.
	if !isScheme(s[:i]) {
		return "", false
	}

	// Scan host part. Must have at least one byte,
	// and must start and end in non-punctuation.
	i += 3
	if i >= len(s) || !isHost(s[i]) || isPunct(s[i]) {
		return "", false
	}
	i++
	end := i
	for i < len(s) && isHost(s[i]) {
		if !isPunct(s[i]) {
			end = i + 1
		}
		i++
	}
	i = end

	// At this point we are definitely returning a URL (scheme://host).
	// We just have to find the longest path we can add to it.
	// Heuristics abound.
	// We allow parens, braces, and brackets,
	// but only if they match (#5043, #22285).
	// We allow .,:;?! in the path but not at the end,
	// to avoid end-of-sentence punctuation (#18139, #16565).
	stk := []byte{}
	end = i
Path:
	for ; i < len(s); i++ {
		if isPunct(s[i]) {
			continue
		}
		if !isPath(s[i]) {
			break
		}
		switch s[i] {
		case '(':
			stk = append(stk, ')')
		case '{':
			stk = append(stk, '}')
		case '[':
			stk = append(stk, ']')
		case ')', '}', ']':
			if len(stk) == 0 || stk[len(stk)-1] != s[i] {
				break Path
			}
			stk = stk[:len(stk)-1]
		}
		if len(stk) == 0 {
			end = i + 1
		}
	}

	return s[:end], true
}

// isScheme reports whether s is a recognized URL scheme.
// Note that if strings of new length (beyond 3-7)
// are added here, the fast path at the top of autoURL will need updating.
func isScheme(s string) bool {
	switch s {
	case "file",
		"ftp",
		"gopher",
		"http",
		"https",
		"mailto",
		"nntp":
		return true
	}
	return false
}

// isHost reports whether c is a byte that can appear in a URL host,
// like www.example.com or user@[::1]:8080
func isHost(c byte) bool {
	// mask is a 128-bit bitmap with 1s for allowed bytes,
	// so that the byte c can be tested with a shift and an and.
	// If c > 128, then 1<<c and 1<<(c-64) will both be zero,
	// and this function will return false.
	const mask = 0 |
		(1<<26-1)<<'A' |
		(1<<26-1)<<'a' |
		(1<<10-1)<<'0' |
		1<<'_' |
		1<<'@' |
		1<<'-' |
		1<<'.' |
		1<<'[' |
		1<<']' |
		1<<':'

	return ((uint64(1)<<c)&(mask&(1<<64-1)) |
		(uint64(1)<<(c-64))&(mask>>64)) != 0
}

// isPunct reports whether c is a punctuation byte that can appear
// inside a path but not at the end.
func isPunct(c byte) bool {
	// mask is a 128-bit bitmap with 1s for allowed bytes,
	// so that the byte c can be tested with a shift and an and.
	// If c > 128, then 1<<c and 1<<(c-64) will both be zero,
	// and this function will return false.
	const mask = 0 |
		1<<'.' |
		1<<',' |
		1<<':' |
		1<<';' |
		1<<'?' |
		1<<'!'

	return ((uint64(1)<<c)&(mask&(1<<64-1)) |
		(uint64(1)<<(c-64))&(mask>>64)) != 0
}

// isPath reports whether c is a (non-punctuation) path byte.
func isPath(c byte) bool {
	// mask is a 128-bit bitmap with 1s for allowed bytes,
	// so that the byte c can be tested with a shift and an and.
	// If c > 128, then 1<<c and 1<<(c-64) will both be zero,
	// and this function will return false.
	const mask = 0 |
		(1<<26-1)<<'A' |
		(1<<26-1)<<'a' |
		(1<<10-1)<<'0' |
		1<<'$' |
		1<<'\'' |
		1<<'(' |
		1<<')' |
		1<<'*' |
		1<<'+' |
		1<<'&' |
		1<<'#' |
		1<<'=' |
		1<<'@' |
		1<<'~' |
		1<<'_' |
		1<<'/' |
		1<<'-' |
		1<<'[' |
		1<<']' |
		1<<'{' |
		1<<'}' |
		1<<'%'

	return ((uint64(1)<<c)&(mask&(1<<64-1)) |
		(uint64(1)<<(c-64))&(mask>>64)) != 0
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
		r, nr := utf8.DecodeRuneInString(s)
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
	const mask = 0 |
		(1<<26-1)<<'A' |
		(1<<26-1)<<'a' |
		(1<<10-1)<<'0' |
		1<<'_'

	return ((uint64(1)<<c)&(mask&(1<<64-1)) |
		(uint64(1)<<(c-64))&(mask>>64)) != 0
}
