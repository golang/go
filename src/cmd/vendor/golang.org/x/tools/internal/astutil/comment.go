// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import (
	"go/ast"
	"go/token"
	"strings"
)

// Deprecation returns the paragraph of the doc comment that starts with the
// conventional "Deprecation: " marker, as defined by
// https://go.dev/wiki/Deprecated, or "" if the documented symbol is not
// deprecated.
func Deprecation(doc *ast.CommentGroup) string {
	for _, p := range strings.Split(doc.Text(), "\n\n") {
		// There is still some ambiguity for deprecation message. This function
		// only returns the paragraph introduced by "Deprecated: ". More
		// information related to the deprecation may follow in additional
		// paragraphs, but the deprecation message should be able to stand on
		// its own. See golang/go#38743.
		if strings.HasPrefix(p, "Deprecated: ") {
			return p
		}
	}
	return ""
}

// -- plundered from the future (CL 605517, issue #68021) --

// TODO(adonovan): replace with ast.Directive after go1.25 (#68021).
// Beware of our local mods to handle analysistest
// "want" comments on the same line.

// A directive is a comment line with special meaning to the Go
// toolchain or another tool. It has the form:
//
//	//tool:name args
//
// The "tool:" portion is missing for the three directives named
// line, extern, and export.
//
// See https://go.dev/doc/comment#Syntax for details of Go comment
// syntax and https://pkg.go.dev/cmd/compile#hdr-Compiler_Directives
// for details of directives used by the Go compiler.
type Directive struct {
	Pos  token.Pos // of preceding "//"
	Tool string
	Name string
	Args string // may contain internal spaces
}

// isDirective reports whether c is a comment directive.
// This code is also in go/printer.
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

// Directives returns the directives within the comment.
func Directives(g *ast.CommentGroup) (res []*Directive) {
	if g != nil {
		// Avoid (*ast.CommentGroup).Text() as it swallows directives.
		for _, c := range g.List {
			if len(c.Text) > 2 &&
				c.Text[1] == '/' &&
				c.Text[2] != ' ' &&
				isDirective(c.Text[2:]) {

				tool, nameargs, ok := strings.Cut(c.Text[2:], ":")
				if !ok {
					// Must be one of {line,extern,export}.
					tool, nameargs = "", tool
				}
				name, args, _ := strings.Cut(nameargs, " ") // tab??
				// Permit an additional line comment after the args, chiefly to support
				// [golang.org/x/tools/go/analysis/analysistest].
				args, _, _ = strings.Cut(args, "//")
				res = append(res, &Directive{
					Pos:  c.Slash,
					Tool: tool,
					Name: name,
					Args: strings.TrimSpace(args),
				})
			}
		}
	}
	return
}
