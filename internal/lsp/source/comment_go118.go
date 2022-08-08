// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package source

// Starting with go1.19, the formatting of comments has changed, and there
// is a new package (go/doc/comment) for processing them.
// As long as gopls has to compile under earlier versions, tests
// have to pass with both the old and new code, which produce
// slightly different results. (cmd/test/definition.go, source/comment_test.go,
// and source/source_test.go) Each of the test files checks the results
// with a function, tests.CheckSameMarkdown, that accepts both the old and the new
// results. (The old code escapes many characters the new code does not,
// and the new code sometimes adds a blank line.)

// When gopls no longer needs to compile with go1.18, the old comment.go should
// be replaced by this file, the golden test files should be updated.
// (and checkSameMarkdown() could be replaced by a simple comparison.)

import "go/doc/comment"

// CommentToMarkdown converts comment text to formatted markdown.
// The comment was prepared by DocReader,
// so it is known not to have leading, trailing blank lines
// nor to have trailing spaces at the end of lines.
// The comment markers have already been removed.
func CommentToMarkdown(text string) string {
	var p comment.Parser
	doc := p.Parse(text)
	var pr comment.Printer
	easy := pr.Markdown(doc)
	return string(easy)
}
