// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmdtest

import (
	"strings"
	"testing"

	"golang.org/x/tools/internal/span"
)

func (r *runner) SemanticTokens(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()
	got, stderr := r.NormalizeGoplsCmd(t, "semtok", filename)
	if stderr != "" {
		t.Fatalf("%s: %q", filename, stderr)
	}
	want := string(r.data.Golden("semantic", filename, func() ([]byte, error) {
		return []byte(got), nil
	}))
	if want != got {
		lwant := strings.Split(want, "\n")
		lgot := strings.Split(got, "\n")
		t.Errorf("want(%d-%d) != got(%d-%d) for %s", len(want), len(lwant), len(got), len(lgot), r.Normalize(filename))
		for i := 0; i < len(lwant) && i < len(lgot); i++ {
			if lwant[i] != lgot[i] {
				t.Errorf("line %d:\nwant%q\ngot %q\n", i, lwant[i], lgot[i])
			}
		}
	}
}
