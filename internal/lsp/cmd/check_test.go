// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

func (r *runner) Diagnostics(t *testing.T, data tests.Diagnostics) {
	for uri, want := range data {
		if len(want) == 1 && want[0].Message == "" {
			continue
		}
		fname := uri.Filename()
		args := []string{"-remote=internal", "check", fname}
		out := captureStdOut(t, func() {
			tool.Main(context.Background(), r.app, args)
		})
		// parse got into a collection of reports
		got := map[string]struct{}{}
		for _, l := range strings.Split(out, "\n") {
			if len(l) == 0 {
				continue
			}
			// parse and reprint to normalize the span
			bits := strings.SplitN(l, ": ", 2)
			if len(bits) == 2 {
				spn := span.Parse(strings.TrimSpace(bits[0]))
				spn = span.New(spn.URI(), spn.Start(), span.Point{})
				l = fmt.Sprintf("%s: %s", spn, strings.TrimSpace(bits[1]))
			}
			got[l] = struct{}{}
		}
		for _, diag := range want {
			spn := span.New(diag.Span.URI(), diag.Span.Start(), diag.Span.Start())
			expect := fmt.Sprintf("%v: %v", spn, diag.Message)
			_, found := got[expect]
			if !found {
				t.Errorf("missing diagnostic %q", expect)
			} else {
				delete(got, expect)
			}
		}
		for extra, _ := range got {
			t.Errorf("extra diagnostic %q", extra)
		}
	}
}
