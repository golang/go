// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"testing"

	"golang.org/x/tools/internal/span"
)

func (r *runner) FoldingRanges(t *testing.T, spn span.Span) {
	goldenTag := "foldingRange-cmd"
	uri := spn.URI()
	filename := uri.Filename()
	got, _ := r.NormalizeGoplsCmd(t, "folding_ranges", filename)
	expect := string(r.data.Golden(goldenTag, filename, func() ([]byte, error) {
		return []byte(got), nil
	}))

	if expect != got {
		t.Errorf("folding_ranges failed failed for %s expected:\n%s\ngot:\n%s", filename, expect, got)
	}
}
