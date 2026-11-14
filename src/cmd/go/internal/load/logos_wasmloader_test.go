// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"slices"
	"testing"
)

func TestFileWasmExportNames(t *testing.T) {
	for _, tt := range []struct {
		name string
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
			name: "invalid source",
			src:  "package p\n//go:wasmexport exported\nfunc f() {\n",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := fileWasmExportNames([]byte(tt.src)); !slices.Equal(got, tt.want) {
				t.Fatalf("fileWasmExportNames() = %q, want %q", got, tt.want)
			}
		})
	}
}
