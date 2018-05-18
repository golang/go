// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/syntax"
	"reflect"
	"testing"
)

func eq(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestPragmaFields(t *testing.T) {
	var tests = []struct {
		in   string
		want []string
	}{
		{"", []string{}},
		{" \t ", []string{}},
		{`""""`, []string{`""`, `""`}},
		{"  a'b'c  ", []string{"a'b'c"}},
		{"1 2 3 4", []string{"1", "2", "3", "4"}},
		{"\n☺\t☹\n", []string{"☺", "☹"}},
		{`"1 2 "  3  " 4 5"`, []string{`"1 2 "`, `3`, `" 4 5"`}},
		{`"1""2 3""4"`, []string{`"1"`, `"2 3"`, `"4"`}},
		{`12"34"`, []string{`12`, `"34"`}},
		{`12"34 `, []string{`12`}},
	}

	for _, tt := range tests {
		got := pragmaFields(tt.in)
		if !eq(got, tt.want) {
			t.Errorf("pragmaFields(%q) = %v; want %v", tt.in, got, tt.want)
			continue
		}
	}
}

func TestPragcgo(t *testing.T) {
	var tests = []struct {
		in   string
		want []string
	}{
		{`go:cgo_export_dynamic local`, []string{`cgo_export_dynamic`, `local`}},
		{`go:cgo_export_dynamic local remote`, []string{`cgo_export_dynamic`, `local`, `remote`}},
		{`go:cgo_export_dynamic local' remote'`, []string{`cgo_export_dynamic`, `local'`, `remote'`}},
		{`go:cgo_export_static local`, []string{`cgo_export_static`, `local`}},
		{`go:cgo_export_static local remote`, []string{`cgo_export_static`, `local`, `remote`}},
		{`go:cgo_export_static local' remote'`, []string{`cgo_export_static`, `local'`, `remote'`}},
		{`go:cgo_import_dynamic local`, []string{`cgo_import_dynamic`, `local`}},
		{`go:cgo_import_dynamic local remote`, []string{`cgo_import_dynamic`, `local`, `remote`}},
		{`go:cgo_import_dynamic local remote "library"`, []string{`cgo_import_dynamic`, `local`, `remote`, `library`}},
		{`go:cgo_import_dynamic local' remote' "lib rary"`, []string{`cgo_import_dynamic`, `local'`, `remote'`, `lib rary`}},
		{`go:cgo_import_static local`, []string{`cgo_import_static`, `local`}},
		{`go:cgo_import_static local'`, []string{`cgo_import_static`, `local'`}},
		{`go:cgo_dynamic_linker "/path/"`, []string{`cgo_dynamic_linker`, `/path/`}},
		{`go:cgo_dynamic_linker "/p ath/"`, []string{`cgo_dynamic_linker`, `/p ath/`}},
		{`go:cgo_ldflag "arg"`, []string{`cgo_ldflag`, `arg`}},
		{`go:cgo_ldflag "a rg"`, []string{`cgo_ldflag`, `a rg`}},
	}

	var p noder
	var nopos syntax.Pos
	for _, tt := range tests {
		p.pragcgobuf = nil
		p.pragcgo(nopos, tt.in)

		got := p.pragcgobuf
		want := [][]string{tt.want}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("pragcgo(%q) = %q; want %q", tt.in, got, want)
			continue
		}
	}
}
