// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"fmt"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

var renameModes = [][]string{
	[]string{},
	[]string{"-d"},
}

func (r *runner) Rename(t *testing.T, spn span.Span, newText string) {
	filename := spn.URI().Filename()
	for _, mode := range renameModes {
		goldenTag := newText + strings.Join(mode, "") + "-rename"
		app := cmd.New("gopls-test", r.data.Config.Dir, r.data.Config.Env)
		loc := fmt.Sprintf("%v", spn)
		args := []string{"-remote=internal", "rename"}
		if strings.Join(mode, "") != "" {
			args = append(args, strings.Join(mode, ""))
		}
		args = append(args, loc, newText)
		var err error
		got := CaptureStdOut(t, func() {
			err = tool.Run(r.ctx, app, args)
		})
		if err != nil {
			got = err.Error()
		}
		got = normalizePaths(r.data, got)
		expect := string(r.data.Golden(goldenTag, filename, func() ([]byte, error) {
			return []byte(got), nil
		}))
		if expect != got {
			t.Errorf("rename failed with %#v expected:\n%s\ngot:\n%s", args, expect, got)
		}
	}
}
