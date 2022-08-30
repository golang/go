// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.19
// +build !go1.19

package tests

import (
	"regexp"
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
)

// The markdown in the golden files matches the converter in comment.go,
// but for go1.19 and later the conversion is done using go/doc/comment.
// Compared to the newer version, the older version
// has extra escapes, and treats code blocks slightly differently.
func CheckSameMarkdown(t *testing.T, got, want string) {
	t.Helper()

	got = normalizeMarkdown(got)
	want = normalizeMarkdown(want)

	if diff := compare.Text(want, got); diff != "" {
		t.Errorf("normalized markdown differs:\n%s", diff)
	}
}

// normalizeMarkdown normalizes whitespace and escaping of the input string, to
// eliminate differences between the Go 1.18 and Go 1.19 generated markdown for
// doc comments. Note that it does not normalize to either the 1.18 or 1.19
// formatting: it simplifies both so that they may be compared.
//
// This function may need to be adjusted as we encounter more differences in
// the generated text.
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
		`\"`, `"`,
		`\.`, `.`,
		`\-`, `-`,
		`\'`, `'`,
		`\n\n\n`, `\n\n`, // Note that these are *escaped* newlines.
	).Replace(input)

	return input
}
