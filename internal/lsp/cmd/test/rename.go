// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"fmt"
	"testing"

	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

func (r *runner) Rename(t *testing.T, spn span.Span, newText string) {
	filename := spn.URI().Filename()
	goldenTag := newText + "-rename"
	app := cmd.New("gopls-test", r.data.Config.Dir, r.data.Config.Env, r.options)
	loc := fmt.Sprintf("%v", spn)
	var err error
	got := CaptureStdOut(t, func() {
		err = tool.Run(r.ctx, app, []string{"-remote=internal", "rename", loc, newText})
	})
	if err != nil {
		got = err.Error()
	}
	got = normalizePaths(r.data, got)
	expect := string(r.data.Golden(goldenTag, filename, func() ([]byte, error) {
		return []byte(got), nil
	}))
	if expect != got {
		t.Errorf("rename failed with %v %v expected:\n%s\ngot:\n%s", loc, newText, expect, got)
	}
	// now check we can build a valid unified diff
	unified := CaptureStdOut(t, func() {
		_ = tool.Run(r.ctx, app, []string{"-remote=internal", "rename", "-d", loc, newText})
	})
	checkUnified(t, filename, expect, unified)
}
