// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/src"
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
		want string
	}{
		{`go:cgo_export_dynamic local`, "cgo_export_dynamic local\n"},
		{`go:cgo_export_dynamic local remote`, "cgo_export_dynamic local remote\n"},
		{`go:cgo_export_dynamic local' remote'`, "cgo_export_dynamic 'local''' 'remote'''\n"},
		{`go:cgo_export_static local`, "cgo_export_static local\n"},
		{`go:cgo_export_static local remote`, "cgo_export_static local remote\n"},
		{`go:cgo_export_static local' remote'`, "cgo_export_static 'local''' 'remote'''\n"},
		{`go:cgo_import_dynamic local`, "cgo_import_dynamic local\n"},
		{`go:cgo_import_dynamic local remote`, "cgo_import_dynamic local remote\n"},
		{`go:cgo_import_dynamic local remote "library"`, "cgo_import_dynamic local remote library\n"},
		{`go:cgo_import_dynamic local' remote' "lib rary"`, "cgo_import_dynamic 'local''' 'remote''' 'lib rary'\n"},
		{`go:cgo_import_static local`, "cgo_import_static local\n"},
		{`go:cgo_import_static local'`, "cgo_import_static 'local'''\n"},
		{`go:cgo_dynamic_linker "/path/"`, "cgo_dynamic_linker /path/\n"},
		{`go:cgo_dynamic_linker "/p ath/"`, "cgo_dynamic_linker '/p ath/'\n"},
		{`go:cgo_ldflag "arg"`, "cgo_ldflag arg\n"},
		{`go:cgo_ldflag "a rg"`, "cgo_ldflag 'a rg'\n"},
	}

	var p noder
	for _, tt := range tests {
		got := p.pragcgo(src.NoPos, tt.in)
		if got != tt.want {
			t.Errorf("pragcgo(%q) = %q; want %q", tt.in, got, tt.want)
			continue
		}
	}
}
