// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd_test

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

var renameModes = [][]string{
	[]string{},
	[]string{"-d"},
}

func (r *runner) Rename(t *testing.T, data tests.Renames) {
	sortedSpans := sortSpans(data) // run the tests in a repeatable order
	for _, spn := range sortedSpans {
		tag := data[spn]
		filename := spn.URI().Filename()
		for _, mode := range renameModes {
			goldenTag := data[spn] + strings.Join(mode, "") + "-rename"
			expect := string(r.data.Golden(goldenTag, filename, func() ([]byte, error) {
				return []byte{}, nil
			}))

			app := cmd.New("gopls-test", r.data.Config.Dir, r.data.Config.Env)
			loc := fmt.Sprintf("%v", spn)
			args := []string{"-remote=internal", "rename"}
			if strings.Join(mode, "") != "" {
				args = append(args, strings.Join(mode, ""))
			}
			args = append(args, loc, tag)
			var err error
			got := captureStdOut(t, func() {
				err = tool.Run(r.ctx, app, args)
			})
			if err != nil {
				got = err.Error()
			}
			got = normalizePaths(r.data, got)
			if expect != got {
				t.Errorf("rename failed with %#v expected:\n%s\ngot:\n%s", args, expect, got)
			}
		}
	}
}

func sortSpans(data map[span.Span]string) []span.Span {
	spans := make([]span.Span, 0, len(data))
	for spn, _ := range data {
		spans = append(spans, spn)
	}
	sort.Slice(spans, func(i, j int) bool {
		return span.Compare(spans[i], spans[j]) < 0
	})
	return spans
}
