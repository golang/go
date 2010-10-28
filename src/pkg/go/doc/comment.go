// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Godoc comment extraction and comment -> HTML formatting.

package doc

import (
	"go/ast"
	"io"
	"regexp"
	"strings"
	"template" // for htmlEscape
)


func isWhitespace(ch byte) bool { return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' }


func stripTrailingWhitespace(s string) string {
	i := len(s)
	for i > 0 && isWhitespace(s[i-1]) {
		i--
	}
	return s[0:i]
}


// CommentText returns the text of comment,
// with the comment markers - //, /*, and */ - removed.
func CommentText(comment *ast.CommentGroup) string {
	if comment == nil {
		return ""
	}
	comments := make([]string, len(comment.List))
	for i, c := range comment.List {
		comments[i] = string(c.Text)
	}

	lines := make([]string, 0, 10) // most comments are less than 10 lines
	for _, c := range comments {
		// Remove comment markers.
		// The parser has given us exactly the comment text.
		switch c[1] {
		case '/':
			//-style comment
			c = c[2:]
			// Remove leading space after //, if there is one.
			// TODO(gri) This appears to be necessary in isolated
			//           cases (bignum.RatFromString) - why?
			if len(c) > 0 && c[0] == ' ' {
				c = c[1:]
			}
		case '*':
			/*-style comment */
			c = c[2 : len(c)-2]
		}

		// Split on newlines.
		cl := strings.Split(c, "\n", -1)

		// Walk lines, stripping trailing white space and adding to list.
		for _, l := range cl {
			lines = append(lines, stripTrailingWhitespace(l))
		}
	}

	// Remove leading blank lines; convert runs of
	// interior blank lines to a single blank line.
	n := 0
	for _, line := range lines {
		if line != "" || n > 0 && lines[n-1] != "" {
			lines[n] = line
			n++
		}
	}
	lines = lines[0:n]

	// Add final "" entry to get trailing newline from Join.
	if n > 0 && lines[n-1] != "" {
		lines = append(lines, "")
	}

	return strings.Join(lines, "\n")
}


// Split bytes into lines.
func split(text []byte) [][]byte {
	// count lines
	n := 0
	last := 0
	for i, c := range text {
		if c == '\n' {
			last = i + 1
			n++
		}
	}
	if last < len(text) {
		n++
	}

	// split
	out := make([][]byte, n)
	last = 0
	n = 0
	for i, c := range text {
		if c == '\n' {
			out[n] = text[last : i+1]
			last = i + 1
			n++
		}
	}
	if last < len(text) {
		out[n] = text[last:]
	}

	return out
}


var (
	ldquo = []byte("&ldquo;")
	rdquo = []byte("&rdquo;")
)

// Escape comment text for HTML. If nice is set,
// also turn `` into &ldquo; and '' into &rdquo;.
func commentEscape(w io.Writer, s []byte, nice bool) {
	last := 0
	if nice {
		for i := 0; i < len(s)-1; i++ {
			ch := s[i]
			if ch == s[i+1] && (ch == '`' || ch == '\'') {
				template.HTMLEscape(w, s[last:i])
				last = i + 2
				switch ch {
				case '`':
					w.Write(ldquo)
				case '\'':
					w.Write(rdquo)
				}
				i++ // loop will add one more
			}
		}
	}
	template.HTMLEscape(w, s[last:])
}


const (
	// Regexp for Go identifiers
	identRx = `[a-zA-Z_][a-zA-Z_0-9]*` // TODO(gri) ASCII only for now - fix this

	// Regexp for URLs
	protocol = `(https?|ftp|file|gopher|mailto|news|nntp|telnet|wais|prospero):`
	hostPart = `[a-zA-Z0-9_@\-]+`
	filePart = `[a-zA-Z0-9_?%#~&/\-+=]+`
	urlRx    = protocol + `//` + // http://
		hostPart + `([.:]` + hostPart + `)*/?` + // //www.google.com:8080/
		filePart + `([:.,]` + filePart + `)*`
)

var matchRx = regexp.MustCompile(`(` + identRx + `)|(` + urlRx + `)`)

var (
	html_a      = []byte(`<a href="`)
	html_aq     = []byte(`">`)
	html_enda   = []byte("</a>")
	html_i      = []byte("<i>")
	html_endi   = []byte("</i>")
	html_p      = []byte("<p>\n")
	html_endp   = []byte("</p>\n")
	html_pre    = []byte("<pre>")
	html_endpre = []byte("</pre>\n")
)


// Emphasize and escape a line of text for HTML. URLs are converted into links;
// if the URL also appears in the words map, the link is taken from the map (if
// the corresponding map value is the empty string, the URL is not converted
// into a link). Go identifiers that appear in the words map are italicized; if
// the corresponding map value is not the empty string, it is considered a URL
// and the word is converted into a link. If nice is set, the remaining text's
// appearance is improved where it makes sense (e.g., `` is turned into &ldquo;
// and '' into &rdquo;).
func emphasize(w io.Writer, line []byte, words map[string]string, nice bool) {
	for {
		m := matchRx.FindSubmatchIndex(line)
		if m == nil {
			break
		}
		// m >= 6 (two parenthesized sub-regexps in matchRx, 1st one is identRx)

		// write text before match
		commentEscape(w, line[0:m[0]], nice)

		// analyze match
		match := line[m[0]:m[1]]
		url := ""
		italics := false
		if words != nil {
			url, italics = words[string(match)]
		}
		if m[2] < 0 {
			// didn't match against first parenthesized sub-regexp; must be match against urlRx
			if !italics {
				// no alternative URL in words list, use match instead
				url = string(match)
			}
			italics = false // don't italicize URLs
		}

		// write match
		if len(url) > 0 {
			w.Write(html_a)
			template.HTMLEscape(w, []byte(url))
			w.Write(html_aq)
		}
		if italics {
			w.Write(html_i)
		}
		commentEscape(w, match, nice)
		if italics {
			w.Write(html_endi)
		}
		if len(url) > 0 {
			w.Write(html_enda)
		}

		// advance
		line = line[m[1]:]
	}
	commentEscape(w, line, nice)
}


func indentLen(s []byte) int {
	i := 0
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	return i
}


func isBlank(s []byte) bool { return len(s) == 0 || (len(s) == 1 && s[0] == '\n') }


func commonPrefix(a, b []byte) []byte {
	i := 0
	for i < len(a) && i < len(b) && a[i] == b[i] {
		i++
	}
	return a[0:i]
}


func unindent(block [][]byte) {
	if len(block) == 0 {
		return
	}

	// compute maximum common white prefix
	prefix := block[0][0:indentLen(block[0])]
	for _, line := range block {
		if !isBlank(line) {
			prefix = commonPrefix(prefix, line[0:indentLen(line)])
		}
	}
	n := len(prefix)

	// remove
	for i, line := range block {
		if !isBlank(line) {
			block[i] = line[n:]
		}
	}
}


// Convert comment text to formatted HTML.
// The comment was prepared by DocReader,
// so it is known not to have leading, trailing blank lines
// nor to have trailing spaces at the end of lines.
// The comment markers have already been removed.
//
// Turn each run of multiple \n into </p><p>
// Turn each run of indented lines into a <pre> block without indent.
//
// URLs in the comment text are converted into links; if the URL also appears
// in the words map, the link is taken from the map (if the corresponding map
// value is the empty string, the URL is not converted into a link).
//
// Go identifiers that appear in the words map are italicized; if the corresponding
// map value is not the empty string, it is considered a URL and the word is converted
// into a link.
func ToHTML(w io.Writer, s []byte, words map[string]string) {
	inpara := false

	close := func() {
		if inpara {
			w.Write(html_endp)
			inpara = false
		}
	}
	open := func() {
		if !inpara {
			w.Write(html_p)
			inpara = true
		}
	}

	lines := split(s)
	unindent(lines)
	for i := 0; i < len(lines); {
		line := lines[i]
		if isBlank(line) {
			// close paragraph
			close()
			i++
			continue
		}
		if indentLen(line) > 0 {
			// close paragraph
			close()

			// count indented or blank lines
			j := i + 1
			for j < len(lines) && (isBlank(lines[j]) || indentLen(lines[j]) > 0) {
				j++
			}
			// but not trailing blank lines
			for j > i && isBlank(lines[j-1]) {
				j--
			}
			block := lines[i:j]
			i = j

			unindent(block)

			// put those lines in a pre block
			w.Write(html_pre)
			for _, line := range block {
				emphasize(w, line, nil, false) // no nice text formatting
			}
			w.Write(html_endpre)
			continue
		}
		// open paragraph
		open()
		emphasize(w, lines[i], words, true) // nice text formatting
		i++
	}
	close()
}
