// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"fmt"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
	"golang.org/x/tools/gopls/internal/span"
)

func (r *runner) Rename(t *testing.T, spn span.Span, newText string) {
	filename := spn.URI().Filename()
	goldenTag := newText + "-rename"
	loc := fmt.Sprintf("%v", spn)
	got, err := r.NormalizeGoplsCmd(t, "rename", loc, newText)
	got += err
	want := string(r.data.Golden(t, goldenTag, filename, func() ([]byte, error) {
		return []byte(got), nil
	}))
	if diff := compare.Text(want, got); diff != "" {
		t.Errorf("rename failed with %v %v (-want +got):\n%s", loc, newText, diff)
	}
	// now check we can build a valid unified diff
	unified, _ := r.NormalizeGoplsCmd(t, "rename", "-d", loc, newText)
	checkUnified(t, filename, want, unified)
}
