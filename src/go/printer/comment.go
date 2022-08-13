// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package printer

import (
	"go/ast"
	"go/doc/comment"
	"strings"
)

// formatDocComment reformats the doc comment list,
// returning the canonical formatting.
func formatDocComment(list []*ast.Comment) []*ast.Comment {
	// Extract comment text (removing comment markers).
	var kind, text string
	var directives []*ast.Comment
	if len(list) == 1 && strings.HasPrefix(list[0].Text, "/*") {
		kind = "/*"
		text = list[0].Text
		if !strings.Contains(text, "\n") || allStars(text) {
			// Single-line /* .. */ comment in doc comment position,
			// or multiline old-style comment like
			//	/*
			//	 * Comment
			//	 * text here.
			//	 */
			// Should not happen, since it will not work well as a
			// doc comment, but if it does, just ignore:
			// reformatting it will only make the situation worse.
			return list
		}
		text = text[2 : len(text)-2] // cut /* and */
	} else if strings.HasPrefix(list[0].Text, "//") {
		kind = "//"
		var b strings.Builder
		for _, c := range list {
			if !strings.HasPrefix(c.Text, "//") {
				return list
			}
			// Accumulate //go:build etc lines separately.
			if isDirective(c.Text[2:]) {
				directives = append(directives, c)
				continue
			}
			b.WriteString(strings.TrimPrefix(c.Text[2:], " "))
			b.WriteString("\n")
		}
		text = b.String()
	} else {
		// Not sure what this is, so leave alone.
		return list
	}

	if text == "" {
		return list
	}

	// Parse comment and reformat as text.
	var p comment.Parser
	d := p.Parse(text)

	var pr comment.Printer
	text = string(pr.Comment(d))

	// For /* */ comment, return one big comment with text inside.
	slash := list[0].Slash
	if kind == "/*" {
		c := &ast.Comment{
			Slash: slash,
			Text:  "/*\n" + text + "*/",
		}
		return []*ast.Comment{c}
	}

	// For // comment, return sequence of // lines.
	var out []*ast.Comment
	for text != "" {
		var line string
		line, text, _ = strings.Cut(text, "\n")
		if line == "" {
			line = "//"
		} else if strings.HasPrefix(line, "\t") {
			line = "//" + line
		} else {
			line = "// " + line
		}
		out = append(out, &ast.Comment{
			Slash: slash,
			Text:  line,
		})
	}
	if len(directives) > 0 {
		out = append(out, &ast.Comment{
			Slash: slash,
			Text:  "//",
		})
		for _, c := range directives {
			out = append(out, &ast.Comment{
				Slash: slash,
				Text:  c.Text,
			})
		}
	}
	return out
}

// isDirective reports whether c is a comment directive.
// See go.dev/issue/37974.
// This code is also in go/ast.
func isDirective(c string) bool {
	// "//line " is a line directive.
	// "//extern " is for gccgo.
	// "//export " is for cgo.
	// (The // has been removed.)
	if strings.HasPrefix(c, "line ") || strings.HasPrefix(c, "extern ") || strings.HasPrefix(c, "export ") {
		return true
	}

	// "//[a-z0-9]+:[a-z0-9]"
	// (The // has been removed.)
	colon := strings.Index(c, ":")
	if colon <= 0 || colon+1 >= len(c) {
		return false
	}
	for i := 0; i <= colon+1; i++ {
		if i == colon {
			continue
		}
		b := c[i]
		if !('a' <= b && b <= 'z' || '0' <= b && b <= '9') {
			return false
		}
	}
	return true
}

// allStars reports whether text is the interior of an
// old-style /* */ comment with a star at the start of each line.
func allStars(text string) bool {
	for i := 0; i < len(text); i++ {
		if text[i] == '\n' {
			j := i + 1
			for j < len(text) && (text[j] == ' ' || text[j] == '\t') {
				j++
			}
			if j < len(text) && text[j] != '*' {
				return false
			}
		}
	}
	return true
}
