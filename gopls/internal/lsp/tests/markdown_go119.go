// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package tests

import (
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
)

// DiffMarkdown compares two markdown strings produced by parsing go doc
// comments.
//
// For go1.19 and later, markdown conversion is done using go/doc/comment.
// Compared to the newer version, the older version has extra escapes, and
// treats code blocks slightly differently.
func DiffMarkdown(want, got string) string {
	return compare.Text(want, got)
}
