// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compare

import (
	"fmt"

	"golang.org/x/tools/internal/lsp/diff"
)

// Text returns a formatted unified diff of the edits to go from want to
// got, returning "" if and only if want == got.
//
// This function is intended for use in testing, and panics if any error occurs
// while computing the diff. It is not sufficiently tested for production use.
func Text(want, got string) string {
	if want == got {
		return ""
	}

	// Add newlines to avoid verbose newline messages ("No newline at end of file").
	want += "\n"
	got += "\n"

	d, err := diff.NComputeEdits("", want, got)

	// Panic on errors.
	//
	// TODO(rfindley): refactor so that this function doesn't need to panic.
	// Computing diffs should never fail.
	if err != nil {
		panic(fmt.Sprintf("computing edits failed: %v", err))
	}

	diff := diff.ToUnified("want", "got", want, d).String()

	// Defensively assert that we get an actual diff, so that we guarantee the
	// invariant that we return "" if and only if want == got.
	//
	// This is probably unnecessary, but convenient.
	if diff == "" {
		panic("empty diff for non-identical input")
	}

	return diff
}
