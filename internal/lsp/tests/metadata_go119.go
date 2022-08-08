// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package tests

import (
	"testing"
)

// The markdown in the golden files matches the converter in comment.go,
// but for go1.19 and later the conversion is done using go/doc/comment.
// Compared to the newer version, the older version
// has extra escapes, and treats code blocks slightly differently.
func CheckSameMarkdown(t *testing.T, got, want string) {
	t.Helper()
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
