// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"os/exec"
	"testing"

	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

func (r *runner) Import(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()
	args := []string{"imports", filename}
	app := cmd.New("gopls-test", r.data.Config.Dir, r.data.Exported.Config.Env, r.options)
	got := CaptureStdOut(t, func() {
		_ = tool.Run(r.ctx, app, args)
	})
	want := string(r.data.Golden("goimports", filename, func() ([]byte, error) {
		cmd := exec.Command("goimports", filename)
		out, _ := cmd.Output() // ignore error, sometimes we have intentionally ungofmt-able files
		return out, nil
	}))
	if want != got {
		t.Errorf("imports failed for %s, expected:\n%v\ngot:\n%v", filename, want, got)
	}
}
