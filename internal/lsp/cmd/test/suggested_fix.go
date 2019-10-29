// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

func (r *runner) SuggestedFix(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()
	args := []string{"fix", "-a", filename}
	app := cmd.New("gopls-test", r.data.Config.Dir, r.data.Exported.Config.Env, r.options)
	got := CaptureStdOut(t, func() {
		_ = tool.Run(r.ctx, app, args)
	})
	want := string(r.data.Golden("suggestedfix", filename, func() ([]byte, error) {
		return []byte(got), nil
	}))
	if want != got {
		t.Errorf("suggested fixes failed for %s, expected:\n%v\ngot:\n%v", filename, want, got)
	}
}
