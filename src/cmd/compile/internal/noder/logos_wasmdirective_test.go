// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

func TestRewriteLogosWasmDirectives(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want string
	}{
		{
			name: "export",
			src:  "package main\n\n//logos:wasmexport add\nfunc add() int32 { return 0 }\n",
			want: "package main\n\n//go:wasmexport add\nfunc add() int32 { return 0 }\n",
		},
		{
			name: "import",
			src:  "package main\n\n//logos:wasmimport host now\nfunc now() int32\n",
			want: "package main\n\n//go:wasmimport host now\nfunc now() int32\n",
		},
		{
			name: "tab after the name",
			src:  "package main\n//logos:wasmexport\tf\nfunc f() {}\n",
			want: "package main\n//go:wasmexport\tf\nfunc f() {}\n",
		},
		{
			name: "bare name",
			src:  "package main\n//logos:wasmexport\nfunc f() {}\n",
			want: "package main\n//go:wasmexport\nfunc f() {}\n",
		},
		{
			name: "trailing comment after code",
			src:  "package main\nfunc f() {} //logos:wasmexport f\n",
			want: "package main\nfunc f() {} //go:wasmexport f\n",
		},
		{
			name: "every directive of a file",
			src:  "package main\n//logos:wasmexport a\nfunc a() {}\n//logos:wasmimport m b\nfunc b()\n//logos:wasmexport c\nfunc c() {}\n",
			want: "package main\n//go:wasmexport a\nfunc a() {}\n//go:wasmimport m b\nfunc b()\n//go:wasmexport c\nfunc c() {}\n",
		},
		{
			name: "longer name stays a comment",
			src:  "package main\n//logos:wasmexported f\nfunc f() {}\n",
			want: "package main\n//logos:wasmexported f\nfunc f() {}\n",
		},
		{
			name: "block comment stays a comment",
			src:  "package main\n/* //logos:wasmexport f */\nfunc f() {}\n",
			want: "package main\n/* //logos:wasmexport f */\nfunc f() {}\n",
		},
		{
			name: "raw string line stays text",
			src:  "package main\n\nvar s = `\n//logos:wasmexport f\n`\n",
			want: "package main\n\nvar s = `\n//logos:wasmexport f\n`\n",
		},
		{
			name: "interpolated string stays text",
			src:  "package main\n\nvar s = $\"//logos:wasmexport f {x}\"\n",
			want: "package main\n\nvar s = $\"//logos:wasmexport f {x}\"\n",
		},
		{
			name: "no directive",
			src:  "package main\n\nfunc f() {}\n",
			want: "package main\n\nfunc f() {}\n",
		},
		{
			name: "non-ASCII text before the directive",
			src:  "package main\n\nvar _ = \"héllo\" //logos:wasmexport f\nfunc f() {}\n",
			want: "package main\n\nvar _ = \"héllo\" //go:wasmexport f\nfunc f() {}\n",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := string(rewriteLogosWasmDirectives([]byte(c.src), syntax.LogosMode))
			if got != c.want {
				t.Errorf("rewrite =\n%q\nwant\n%q", got, c.want)
			}
		})
	}
}

func TestRewriteLogosWasmDirectivesKeepsLines(t *testing.T) {
	src := "package main\n\n//logos:wasmexport a\nfunc a() {}\n\n//logos:wasmimport m b\nfunc b()\n"
	got := string(rewriteLogosWasmDirectives([]byte(src), syntax.LogosMode))
	if n, m := strings.Count(src, "\n"), strings.Count(got, "\n"); n != m {
		t.Fatalf("line count changed from %d to %d", n, m)
	}
}
