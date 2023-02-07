// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compare

import (
	"bytes"

	"golang.org/x/tools/internal/diff"
)

// Text returns a formatted unified diff of the edits to go from want to
// got, returning "" if and only if want == got.
//
// This function is intended for use in testing, and panics if any error occurs
// while computing the diff. It is not sufficiently tested for production use.
func Text(want, got string) string {
	return NamedText("want", "got", want, got)
}

// NamedText is like text, but allows passing custom names of the 'want' and
// 'got' content.
func NamedText(wantName, gotName, want, got string) string {
	if want == got {
		return ""
	}

	// Add newlines to avoid verbose newline messages ("No newline at end of file").
	unified := diff.Unified(wantName, gotName, want+"\n", got+"\n")

	// Defensively assert that we get an actual diff, so that we guarantee the
	// invariant that we return "" if and only if want == got.
	//
	// This is probably unnecessary, but convenient.
	if unified == "" {
		panic("empty diff for non-identical input")
	}

	return unified
}

// Bytes is like Text but using byte slices.
func Bytes(want, got []byte) string {
	if bytes.Equal(want, got) {
		return "" // common case
	}
	return Text(string(want), string(got))
}
