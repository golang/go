// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"strings"
	"testing"
)

// Upstream records panic calls for result-bearing function termination.
// loGo/s also needs those facts to narrow options in a function with no results.
func TestCompilePanicNarrowingVoid(t *testing.T) {
	for _, tt := range []struct {
		name string
		decl string
		call string
		safe bool
	}{
		{"builtin", "", `panic("nil")`, true},
		{"parenthesized builtin", "", `(panic)("nil")`, true},
		{"shadowed function", `func panic(any) {}`, `panic("nil")`, false},
		{"shadowed variable", `var panic = func(any) {}`, `panic("nil")`, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			src := "package p\n" + tt.decl + `
type payload struct{ n int }
func target(p *payload?) {
	if p == nil { ` + tt.call + ` }
	_ = p.n
}
`
			res := compileFile(t, "x.lgo", src)
			if tt.safe {
				if res.exitCode != 0 || res.output != "" || !res.objectOK {
					t.Errorf("real panic guard: exit %d, output %q, object %v", res.exitCode, res.output, res.objectOK)
				}
				return
			}
			if res.exitCode != 2 || !strings.Contains(res.output, "[correctness::nil_deref_unproven]") || res.objectOK {
				t.Errorf("shadowed panic guard: exit %d, output %q, object %v", res.exitCode, res.output, res.objectOK)
			}
		})
	}
}
