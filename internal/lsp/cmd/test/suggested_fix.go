// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"testing"

	"golang.org/x/tools/internal/span"
)

func (r *runner) SuggestedFix(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()
	got, _ := r.NormalizeGoplsCmd(t, "fix", "-a", filename)
	want := string(r.data.Golden("suggestedfix", filename, func() ([]byte, error) {
		return []byte(got), nil
	}))
	if want != got {
		t.Errorf("suggested fixes failed for %s, expected:\n%v\ngot:\n%v", filename, want, got)
	}
}
