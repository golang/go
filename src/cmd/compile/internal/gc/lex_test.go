// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/syntax"
	"reflect"
	"runtime"
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
	type testStruct struct {
		in   string
		want []string
	}

	var tests = []testStruct{
		{`go:cgo_export_dynamic local`, []string{`cgo_export_dynamic`, `local`}},
		{`go:cgo_export_dynamic local remote`, []string{`cgo_export_dynamic`, `local`, `remote`}},
		{`go:cgo_export_dynamic local' remote'`, []string{`cgo_export_dynamic`, `local'`, `remote'`}},
		{`go:cgo_export_static local`, []string{`cgo_export_static`, `local`}},
		{`go:cgo_export_static local remote`, []string{`cgo_export_static`, `local`, `remote`}},
		{`go:cgo_export_static local' remote'`, []string{`cgo_export_static`, `local'`, `remote'`}},
		{`go:cgo_import_dynamic local`, []string{`cgo_import_dynamic`, `local`}},
		{`go:cgo_import_dynamic local remote`, []string{`cgo_import_dynamic`, `local`, `remote`}},
		{`go:cgo_import_static local`, []string{`cgo_import_static`, `local`}},
		{`go:cgo_import_static local'`, []string{`cgo_import_static`, `local'`}},
		{`go:cgo_dynamic_linker "/path/"`, []string{`cgo_dynamic_linker`, `/path/`}},
		{`go:cgo_dynamic_linker "/p ath/"`, []string{`cgo_dynamic_linker`, `/p ath/`}},
		{`go:cgo_ldflag "arg"`, []string{`cgo_ldflag`, `arg`}},
		{`go:cgo_ldflag "a rg"`, []string{`cgo_ldflag`, `a rg`}},
	}

	if runtime.GOOS != "aix" {
		tests = append(tests, []testStruct{
			{`go:cgo_import_dynamic local remote "library"`, []string{`cgo_import_dynamic`, `local`, `remote`, `library`}},
			{`go:cgo_import_dynamic local' remote' "lib rary"`, []string{`cgo_import_dynamic`, `local'`, `remote'`, `lib rary`}},
		}...)
	} else {
		// cgo_import_dynamic with a library is slightly different on AIX
		// as the library field must follow the pattern [libc.a/object.o].
		tests = append(tests, []testStruct{
			{`go:cgo_import_dynamic local remote "lib.a/obj.o"`, []string{`cgo_import_dynamic`, `local`, `remote`, `lib.a/obj.o`}},
			// This test must fail.
			{`go:cgo_import_dynamic local' remote' "library"`, []string{`<unknown position>: usage: //go:cgo_import_dynamic local [remote ["lib.a/object.o"]]`}},
		}...)

	}

	var p noder
	var nopos syntax.Pos
	for _, tt := range tests {

		p.err = make(chan syntax.Error)
		gotch := make(chan [][]string, 1)
		go func() {
			p.pragcgobuf = nil
			p.pragcgo(nopos, tt.in)
			if p.pragcgobuf != nil {
				gotch <- p.pragcgobuf
			}
		}()

		select {
		case e := <-p.err:
			want := tt.want[0]
			if e.Error() != want {
				t.Errorf("pragcgo(%q) = %q; want %q", tt.in, e, want)
				continue
			}
		case got := <-gotch:
			want := [][]string{tt.want}
			if !reflect.DeepEqual(got, want) {
				t.Errorf("pragcgo(%q) = %q; want %q", tt.in, got, want)
				continue
			}
		}

	}
}
