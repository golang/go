// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"cmp"
	"slices"
	"testing"
)

func TestFileWasmExportNames(t *testing.T) {
	for _, tt := range []struct {
		name string
		file string
		src  string
		want []string
	}{
		{
			name: "function directive",
			src:  "package p\n//go:wasmexport exported\nfunc f() {}\n",
			want: []string{"exported"},
		},
		{
			name: "block comment",
			src:  "package p\n/*\n//go:wasmexport exported\n*/\nfunc f() {}\n",
		},
		{
			name: "raw string",
			src:  "package p\nvar s = `\n//go:wasmexport exported\n`\nfunc f() {}\n",
		},
		{
			name: "two names",
			src:  "package p\n//go:wasmexport first second\nfunc f() {}\n",
		},
		{
			name: "no name",
			src:  "package p\n//go:wasmexport \nfunc f() {}\n",
		},
		{
			name: "indented directive",
			src:  "package p\n \t//go:wasmexport exported\nfunc f() {}\n",
			want: []string{"exported"},
		},
		{
			name: "tab separator",
			src:  "package p\n//go:wasmexport\texported\nfunc f() {}\n",
		},
		{
			name: "variable directive",
			src:  "package p\n//go:wasmexport exported\nvar x int\n",
		},
		{
			name: "code before directive",
			src:  "package p\nvar x int //go:wasmexport exported\nfunc f() {}\n",
		},
		{
			name: "method directive",
			src:  "package p\ntype T int\n//go:wasmexport exported\nfunc (T) f() {}\n",
		},
		{
			name: "multiple functions",
			src:  "package p\n//go:wasmexport first\nfunc f() {}\n//go:wasmexport second\nfunc g() {}\n",
			want: []string{"first", "second"},
		},
		{
			name: "interpolated string",
			src:  "package p\n//go:wasmexport exported\nfunc f() {}\nfunc label() string { return $\"helper\" }\n",
			want: []string{"exported"},
		},
		{
			name: "postfix question",
			src:  "package p\n//go:wasmexport exported\nfunc f() string { data := os.ReadFile(path)?\n\treturn string(data) }\n",
			want: []string{"exported"},
		},
		{
			name: "raw string in hole",
			src:  "package p\nvar s = $\"{f(`\n//go:wasmexport hidden\n`)}\"\nfunc f() {}\n",
		},
		{
			name: "comment in hole",
			src:  "package p\nvar s = $\"{\n//go:wasmexport hidden\n}\"\nfunc f() {}\n",
		},
		{
			name: "literal braces in interpolation",
			src:  "package p\nvar s = $\"{{ //go:wasmexport notreal }}\"\n//go:wasmexport exported\nfunc f() {}\n",
			want: []string{"exported"},
		},
		{
			name: "blank line and comment",
			src:  "package p\n//go:wasmexport exported\n\n// f docs\nfunc f() {}\n",
			want: []string{"exported"},
		},
		{
			name: "two directives",
			src:  "package p\n//go:wasmexport first\n//go:wasmexport second\nfunc f() {}\n",
			want: []string{"second"},
		},
		{
			name: "type after directive",
			src:  "package p\n//go:wasmexport exported\ntype T int\nfunc f() {}\n",
		},
		{
			name: "truncated interpolation",
			src:  "package p\nvar s = $\"{",
		},
		{
			name: "truncated raw string",
			src:  "package p\nvar s = `",
		},
		{
			name: "truncated block comment",
			src:  "package p\nvar /*",
		},
		{
			name: "funcs identifier",
			src:  "package p\n//go:wasmexport exported\nfuncs()\n",
		},
		{
			name: "logos spelling",
			src:  "package p\n//logos:wasmexport exported\nfunc f() string { return $\"x\" }\n",
			want: []string{"exported"},
		},
		{
			name: "mixed spellings",
			src:  "package p\n//logos:wasmexport first\nfunc f() {}\n//go:wasmexport second\nfunc g() {}\n",
			want: []string{"first", "second"},
		},
		{
			name: "logos spelling in a Go file",
			file: "p.go",
			src:  "package p\n//logos:wasmexport exported\nfunc f() {}\n",
		},
		{
			name: "logos spelling with a longer word",
			src:  "package p\n//logos:wasmexported exported\nfunc f() {}\n",
		},
		{
			name: "logos spelling with a tab separator",
			src:  "package p\n//logos:wasmexport\texported\nfunc f() {}\n",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := fileWasmExportNames(cmp.Or(tt.file, "p.lgo"), []byte(tt.src)); !slices.Equal(got, tt.want) {
				t.Fatalf("fileWasmExportNames() = %q, want %q", got, tt.want)
			}
		})
	}
}
