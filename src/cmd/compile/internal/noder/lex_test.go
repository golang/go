// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"internal/buildcfg"
	"reflect"
	"runtime"
	"testing"

	"cmd/compile/internal/syntax"
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
		{`go:cgo_import_dynamic _cgo_panic _cgo_panic,--plugin=/tmp/pwn.so`, []string{`<unknown position>: invalid symbol name "_cgo_panic,--plugin=/tmp/pwn.so" in //go:cgo_import_dynamic`}},
		{`go:cgo_export_dynamic main main,--plugin=/tmp/pwn.so`, []string{`<unknown position>: invalid symbol name "main,--plugin=/tmp/pwn.so" in //go:cgo_export_dynamic`}},
		{`go:cgo_dynamic_linker "/lib64/ld.so,--plugin=/tmp/pwn.so"`, []string{`<unknown position>: invalid path "/lib64/ld.so,--plugin=/tmp/pwn.so" in //go:cgo_dynamic_linker`}},
		{`go:cgo_ldflag "-L\"foo\""`, []string{`cgo_ldflag`, `-L"foo"`}},
		{`go:cgo_ldflag "-L\q\"foo\""`, []string{`<unknown position>: usage: //go:cgo_ldflag "arg"`}},
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

	for _, tt := range tests {
		var p noder
		var nopos syntax.Pos

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

func TestSafeArg(t *testing.T) {
	good := []string{
		"libc.so.6",
		"libstdc++.so.6",
		"libc++.1.dylib",
		"libSystem.B.dylib",
		"kernel32.dll",
		"/usr/lib/libSystem.B.dylib",
		"lib.a/obj.o",
	}
	if runtime.GOOS != "windows" {
		good = append(good, "@rpath/libfoo.dylib")
	}
	for _, s := range good {
		if !safeArg(s) {
			t.Errorf("safeArg(%q) = false, want true", s)
		}
	}

	bad := []string{
		"",
		"-lfoo",
		"../../../../../../tmp/evil.dll",
		"foo/../bar.dll",
		`..\..\evil.dll`,
		":evil.dll",
		"foo bar.dll",
		"foo,bar.so",
	}
	if runtime.GOOS == "windows" {
		bad = append(bad, "@resp")
	}
	for _, s := range bad {
		if safeArg(s) {
			t.Errorf("safeArg(%q) = true, want false", s)
		}
	}
}

func TestPragmaCgoImportDynamic(t *testing.T) {
	origGOOS := buildcfg.GOOS
	defer func() { buildcfg.GOOS = origGOOS }()

	cgoPos := syntax.MakePos(syntax.NewTrimmedFileBase("/tmp/_cgo_import.go", true), 1, 1)
	userPos := syntax.MakePos(syntax.NewTrimmedFileBase("/tmp/main.go", true), 1, 1)
	spoofedPos := syntax.MakePos(syntax.NewLineBase(userPos, "/tmp/_cgo_import.go", true, 1, 1), 2, 1)

	libs := []string{
		"/tmp/r/app/../lib/libfoo.so",
		"$ORIGIN/libfoo.so",
		"/Library/Application Support/Foo/libfoo.dylib",
	}

	check := func(pos syntax.Pos, lib string) error {
		p := noder{err: make(chan syntax.Error, 4)}
		p.pragma(pos, true, `go:cgo_import_dynamic _ _ "`+lib+`"`, nil)
		close(p.err)
		for err := range p.err {
			return err
		}
		return nil
	}

	for _, goos := range []string{"linux", "darwin", "windows"} {
		buildcfg.GOOS = goos
		for _, lib := range libs {
			if err := check(userPos, lib); err == nil {
				t.Errorf("GOOS=%s user file: expected error for library %q, got nil", goos, lib)
			}
			if err := check(spoofedPos, lib); err == nil {
				t.Errorf("GOOS=%s //line-spoofed file: expected error for library %q, got nil", goos, lib)
			}
			err := check(cgoPos, lib)
			if goos == "windows" {
				if err == nil {
					t.Errorf("GOOS=%s _cgo_import.go: expected error for library %q, got nil", goos, lib)
				}
			} else if err != nil {
				t.Errorf("GOOS=%s _cgo_import.go: unexpected error for library %q: %v", goos, lib, err)
			}
		}
	}
}
