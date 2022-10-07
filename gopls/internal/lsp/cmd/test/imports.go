// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"testing"

	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/diff"
)

func (r *runner) Import(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()
	got, _ := r.NormalizeGoplsCmd(t, "imports", filename)
	want := string(r.data.Golden(t, "goimports", filename, func() ([]byte, error) {
		return []byte(got), nil
	}))
	if want != got {
		unified := diff.Unified("want", "got", want, got)
		t.Errorf("imports failed for %s, expected:\n%s", filename, unified)
	}
}
