// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"testing"
)

const assignedCloserHead = `package p

type resource struct{}
func (resource) Close() error { return nil }
func openResource() resource { return resource{} }
`

// A Close call executes before its error result is assigned. Handling that
// result must not make the resource appear unclosed. Calls nested inside a
// conditional expression, errdefer and asynchronous calls are separate cases.
func TestUnclosedCloserAssignedClose(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	tests := []struct {
		name string
		body string
		want int
	}{
		{"assignment", `err := f.Close(); _ = err`, 0},
		{"blank assignment", `_ = f.Close()`, 0},
		{"declaration", `var err = f.Close(); _ = err`, 0},
		{"parenthesized call", `err := (f.Close()); _ = err`, 0},
		{"parallel assignment", `err, n := f.Close(), 1; _, _ = err, n`, 0},
		{"parallel declaration", `var err, n = f.Close(), 1; _, _ = err, n`, 0},
		{"if initializer", `if err := f.Close(); err != nil { panic(err) }`, 0},
		{"for initializer", `for err := f.Close(); err != nil; { break }`, 0},
		{"switch initializer", `switch err := f.Close(); err { default: }`, 0},
		{"stable alias", `g := f; err := g.Close(); _ = err`, 0},
		{"no close", `_ = f`, 1},
		{"conditional and", `_ = cond && f.Close() == nil`, 1},
		{"conditional or", `_ = cond || f.Close() == nil`, 1},
		{"different object", `g := openResource(); _ = g.Close(); _ = f`, 1},
		{"shadowed name", `{ f := openResource(); _ = f.Close() }; _ = f`, 1},
	}
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "x.go"
		if kind == KindLgo {
			name = "x.lgo"
		}
		for _, tt := range tests {
			t.Run(name+"/"+tt.name, func(t *testing.T) {
				src := assignedCloserHead + "\nfunc target(cond bool) {\n f := openResource()\n" + tt.body + "\n}\n"
				in := check(t, kind, name, src)
				got := runRule(in, policyOf(in), collectDirectives(in), unclosedIOCloser)
				if len(got) != tt.want {
					t.Errorf("unclosed results = %v, want %d", summarize(got), tt.want)
				}
			})
		}
	}
	for _, tt := range []struct {
		name string
		kind Kind
		src  string
	}{
		{"async.go", KindGo, `func target() { f := openResource(); go f.Close() }`},
		{"errdefer.lgo", KindLgo, `func target() error { f := openResource(); errdefer f.Close(); return nil }`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			in := check(t, tt.kind, tt.name, assignedCloserHead+tt.src)
			got := runRule(in, policyOf(in), collectDirectives(in), unclosedIOCloser)
			if len(got) != 1 {
				t.Errorf("unclosed results = %v, want one", summarize(got))
			}
		})
	}
}

func TestCompileCloserAssignedClose(t *testing.T) {
	src := assignedCloserHead + `
func target() {
	f := openResource()
	if err := f.Close(); err != nil { panic(err) }
}
`
	for _, name := range []string{"x.go", "x.lgo"} {
		t.Run(name, func(t *testing.T) {
			res := compileFile(t, name, src, "-logos.lintonly")
			if res.exitCode != 0 || res.output != "" || res.objectOK {
				t.Errorf("assigned close: exit %d, output %q, object %v", res.exitCode, res.output, res.objectOK)
			}
		})
	}
}
