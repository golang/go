// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.19
// +build !go1.19

package tests

import (
	"regexp"
	"strings"

	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
)

// DiffMarkdown compares two markdown strings produced by parsing go doc
// comments.
//
// For go1.19 and later, markdown conversion is done using go/doc/comment.
// Compared to the newer version, the older version has extra escapes, and
// treats code blocks slightly differently.
func DiffMarkdown(want, got string) string {
	want = normalizeMarkdown(want)
	got = normalizeMarkdown(got)
	return compare.Text(want, got)
}

// normalizeMarkdown normalizes whitespace and escaping of the input string, to
// eliminate differences between the Go 1.18 and Go 1.19 generated markdown for
// doc comments. Note that it does not normalize to either the 1.18 or 1.19
// formatting: it simplifies both so that they may be compared.
//
// This function may need to be adjusted as we encounter more differences in
// the generated text.
//
// TODO(rfindley): this function doesn't correctly handle the case of
// multi-line docstrings.
func normalizeMarkdown(input string) string {
	input = strings.TrimSpace(input)

	// For simplicity, eliminate blank lines.
	input = regexp.MustCompile("\n+").ReplaceAllString(input, "\n")

	// Replace common escaped characters with their unescaped version.
	//
	// This list may not be exhaustive: it was just sufficient to make tests
	// pass.
	input = strings.NewReplacer(
		`\\`, ``,
		`\@`, `@`,
		`\(`, `(`,
		`\)`, `)`,
		`\{`, `{`,
		`\}`, `}`,
		`\"`, `"`,
		`\.`, `.`,
		`\-`, `-`,
		`\'`, `'`,
		`\+`, `+`,
		`\~`, `~`,
		`\=`, `=`,
		`\:`, `:`,
		`\?`, `?`,
		`\n\n\n`, `\n\n`, // Note that these are *escaped* newlines.
	).Replace(input)

	return input
}
