// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import (
	"go/ast"
	"go/token"
	"iter"
	"sort"
	"strings"
)

// Deprecation returns the paragraph of the doc comment that starts with the
// conventional "Deprecation: " marker, or the end of a single-line comment
// with the deprecation marker, as defined by https://go.dev/wiki/Deprecated.
// Returns "" if the documented symbol is not deprecated.
//
// Deprecation(nil) returns the empty string.
func Deprecation(doc *ast.CommentGroup) string {
	// doc.Text() is newline-terminated. For legacy reasons, this function will
	// return as newline-terminated if is the last segment of the CommentGroup
	// but not if it is a paragraph in the middle of the CommentGroup.
	docText := doc.Text()
	for p := range strings.SplitSeq(docText, "\n\n") {
		// There is still some ambiguity for deprecation message. This function
		// only returns the paragraph introduced by "Deprecated: ". More
		// information related to the deprecation may follow in additional
		// paragraphs, but the deprecation message should be able to stand on
		// its own. See golang/go#38743.
		if strings.HasPrefix(p, "Deprecated: ") {
			return p
		}
	}

	// We also want to support deprecation markers in line comments. Not all
	// call sites know whether they have a line comment or the type of AST node
	// the comment is associated with; so to best match line deprecations,
	// the CommentGroup must meet these criteria:
	//   * The doc.Text() is a single line.
	//   * The comment uses the "// ..." format.
	if doc == nil || len(doc.List) != 1 || !strings.HasPrefix(doc.List[0].Text, "//") {
		return ""
	}
	if i := strings.Index(docText, "Deprecated: "); i != -1 {
		return docText[i:]
	}
	return ""
}

// Directives returns the directives within the comment.
//
// It does not report the tool-less directives named line, extern,
// and export.
func Directives(g *ast.CommentGroup) (res []ast.Directive) {
	if g != nil {
		// Avoid (*ast.CommentGroup).Text() as it swallows directives.
		for _, c := range g.List {
			if d, ok := ast.ParseDirective(c.Slash, c.Text); ok {
				res = append(res, d)
			}
		}
	}
	return
}

// Comments returns an iterator over the comments overlapping the specified interval.
// Comments are sorted by position in the file, so we can use binary search.
func Comments(file *ast.File, start, end token.Pos) iter.Seq[*ast.Comment] {
	return func(yield func(*ast.Comment) bool) {
		// Find the first comment group that overlaps the range.
		i := sort.Search(len(file.Comments), func(i int) bool {
			return file.Comments[i].End() >= start
		})
		for _, cg := range file.Comments[i:] {
			if cg.Pos() > end {
				return
			}
			// Find the first comment in the group that overlaps the range.
			j := sort.Search(len(cg.List), func(j int) bool {
				return cg.List[j].End() >= start
			})
			for _, co := range cg.List[j:] {
				if co.Pos() > end {
					return
				}
				if !yield(co) {
					return
				}
			}
		}
	}
}
