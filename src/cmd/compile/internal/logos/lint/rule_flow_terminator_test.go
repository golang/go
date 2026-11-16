// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"testing"
)

// A call can terminate flow only when its checked object has that contract.
// The spelling panic is not such a contract after a local declaration shadows it.
func TestTypedNilReturnPanicIdentity(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tests := []struct {
		name string
		decl string
		call string
		want int
	}{
		{"builtin", "", `panic("nil")`, 0},
		{"parenthesized builtin", "", `(panic)("nil")`, 0},
		{"shadowed function", `func panic(any) {}`, `panic("nil")`, 1},
		{"shadowed variable", `var panic = func(any) {}`, `panic("nil")`, 1},
		{"parenthesized shadow", `func panic(any) {}`, `(panic)("nil")`, 1},
		{"ordinary returning call", `func log(any) {}`, `log("nil")`, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := "package p\n" + tt.decl + `
type problem struct{}
func (*problem) Error() string { return "problem" }
func target(p *problem?) any {
	if p == nil { ` + tt.call + ` }
	return p
}
`
			in := check(t, KindLgo, "x.lgo", src)
			got := runRule(in, policyOf(in), collectDirectives(in), typedNilReturn)
			if len(got) != tt.want {
				t.Errorf("typed nil results = %v, want %d", summarize(got), tt.want)
			}
		})
	}
}

func TestInfiniteRecursionPanicIdentity(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tests := []struct {
		name string
		decl string
		call string
		want int
	}{
		{"builtin", "", `panic("stop")`, 0},
		{"parenthesized builtin", "", `(panic)("stop")`, 0},
		{"shadowed function", `func panic(any) {}`, `panic("stop")`, 1},
		{"shadowed variable", `var panic = func(any) {}`, `panic("stop")`, 1},
		{"parenthesized shadow", `func panic(any) {}`, `(panic)("stop")`, 1},
		{"ordinary returning call", `func log(any) {}`, `log("stop")`, 1},
	}
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "x.go"
		if kind == KindLgo {
			name = "x.lgo"
		}
		for _, tt := range tests {
			t.Run(name+"/"+tt.name, func(t *testing.T) {
				src := "package p\n" + tt.decl + "\nfunc target(n int) int { " + tt.call + "; return target(n) }\n"
				in := check(t, kind, name, src)
				got := runRule(in, policyOf(in), collectDirectives(in), infiniteRecursion)
				if len(got) != tt.want {
					t.Errorf("recursion results = %v, want %d", summarize(got), tt.want)
				}
			})
		}
	}
}
