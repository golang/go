// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"fmt"
	"testing"

	"golang.org/x/tools/internal/span"
)

func (r *runner) Rename(t *testing.T, spn span.Span, newText string) {
	filename := spn.URI().Filename()
	goldenTag := newText + "-rename"
	loc := fmt.Sprintf("%v", spn)
	got, err := r.NormalizeGoplsCmd(t, "rename", loc, newText)
	got += err
	expect := string(r.data.Golden(goldenTag, filename, func() ([]byte, error) {
		return []byte(got), nil
	}))
	if expect != got {
		t.Errorf("rename failed with %v %v\nexpected:\n%s\ngot:\n%s", loc, newText, expect, got)
	}
	// now check we can build a valid unified diff
	unified, _ := r.NormalizeGoplsCmd(t, "rename", "-d", loc, newText)
	checkUnified(t, filename, expect, unified)
}
