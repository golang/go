// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"io/ioutil"
	"strings"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/tests"
	"golang.org/x/tools/gopls/internal/span"
)

// Diagnostics runs the gopls command on a single file, parses its
// diagnostics, and compares against the expectations defined by
// markers in the source file.
func (r *runner) Diagnostics(t *testing.T, uri span.URI, want []*source.Diagnostic) {
	out, _ := r.runGoplsCmd(t, "check", uri.Filename())

	content, err := ioutil.ReadFile(uri.Filename())
	if err != nil {
		t.Fatal(err)
	}
	mapper := protocol.NewColumnMapper(uri, content)

	// Parse command output into a set of diagnostics.
	var got []*source.Diagnostic
	for _, line := range strings.Split(out, "\n") {
		if line == "" {
			continue // skip blank
		}
		parts := strings.SplitN(line, ": ", 2) // "span: message"
		if len(parts) != 2 {
			t.Fatalf("output line not of form 'span: message': %q", line)
		}
		spn, message := span.Parse(parts[0]), parts[1]
		rng, err := mapper.Range(spn)
		if err != nil {
			t.Fatal(err)
		}
		// Set only the fields needed by DiffDiagnostics.
		got = append(got, &source.Diagnostic{
			URI:     uri,
			Range:   rng,
			Message: message,
		})
	}

	// Don't expect fields that we can't populate from the command output.
	for _, diag := range want {
		if diag.Source == "no_diagnostics" {
			continue // see DiffDiagnostics
		}
		diag.Source = ""
		diag.Severity = 0
	}

	tests.CompareDiagnostics(t, uri, want, got)
}
