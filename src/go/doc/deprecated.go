// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import "strings"

// Deprecated returns the deprecation notice, if any, in the doc comment text doc.
// A deprecation notice is the text of a paragraph that starts with "Deprecated: ",
// stripped of the "Deprecated: " prefix.
// If there is no deprecation notice, it returns "", false.
//
// The doc argument is comment text as returned by [go/ast.CommentGroup.Text],
// such as [Func.Doc], [Package.Doc], [Type.Doc], or [Value.Doc].
func Deprecated(doc string) (text string, ok bool) {
	// Comments not at column 1 (struct fields, grouped specs) keep their
	// indentation after gofmt, so strip it as go/doc/comment does.
	indent := commonIndent(doc)
	var lines []string
	prevBreak := true
	for line := range strings.Lines(doc) {
		line = strings.TrimPrefix(strings.TrimSuffix(line, "\n"), indent)
		// As in go/doc/comment, a paragraph ends at a blank or indented line.
		brk := line == "" || line[0] == ' ' || line[0] == '\t'
		if lines != nil {
			if brk {
				break
			}
			lines = append(lines, line)
		} else if prevBreak {
			if rest, ok := strings.CutPrefix(line, "Deprecated: "); ok {
				lines = append(lines, rest)
			}
		}
		prevBreak = brk
	}
	if lines == nil {
		return "", false
	}
	return strings.Join(lines, "\n"), true
}

// commonIndent returns the longest leading space/tab prefix shared by all
// non-blank lines in doc.
func commonIndent(doc string) string {
	var prefix string
	first := true
	for line := range strings.SplitSeq(doc, "\n") {
		trimmed := strings.TrimLeft(line, " \t")
		if trimmed == "" {
			continue
		}
		sp := line[:len(line)-len(trimmed)]
		if first {
			prefix, first = sp, false
		} else {
			i := 0
			for i < len(prefix) && i < len(sp) && prefix[i] == sp[i] {
				i++
			}
			prefix = prefix[:i]
		}
		if prefix == "" {
			return ""
		}
	}
	return prefix
}
