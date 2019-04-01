// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/cmd"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

type diagnostics map[string][]source.Diagnostic

func (l diagnostics) collect(spn span.Span, msgSource, msg string) {
	fname, err := spn.URI().Filename()
	if err != nil {
		return
	}
	//TODO: diagnostics with range
	spn = span.New(spn.URI(), spn.Start(), span.Point{})
	l[fname] = append(l[fname], source.Diagnostic{
		Span:     spn,
		Message:  msg,
		Source:   msgSource,
		Severity: source.SeverityError,
	})
}

func (l diagnostics) test(t *testing.T, e *packagestest.Exported) {
	count := 0
	for fname, want := range l {
		if len(want) == 1 && want[0].Message == "" {
			continue
		}
		args := []string{"-remote=internal"}
		args = append(args, "check", fname)
		app := &cmd.Application{}
		app.Config = *e.Config
		out := captureStdOut(t, func() {
			tool.Main(context.Background(), app, args)
		})
		// parse got into a collection of reports
		got := map[string]struct{}{}
		for _, l := range strings.Split(out, "\n") {
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
			expect := fmt.Sprintf("%v: %v", diag.Span, diag.Message)
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
		count += len(want)
	}
	if count != expectedDiagnosticsCount {
		t.Errorf("got %v diagnostics expected %v", count, expectedDiagnosticsCount)
	}
}
