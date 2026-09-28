// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"cmd/internal/bio"
	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
)

func TestDedupLibraries(t *testing.T) {
	ctxt := &Link{}
	ctxt.Target.HeadType = objabi.Hlinux

	libs := []string{"libc.so", "libc.so.6"}

	got := dedupLibraries(ctxt, libs)
	if !reflect.DeepEqual(got, libs) {
		t.Errorf("dedupLibraries(%v) = %v, want %v", libs, got, libs)
	}
}

func TestDedupLibrariesOpenBSD(t *testing.T) {
	ctxt := &Link{}
	ctxt.Target.HeadType = objabi.Hopenbsd

	tests := []struct {
		libs []string
		want []string
	}{
		{
			libs: []string{"libc.so"},
			want: []string{"libc.so"},
		},
		{
			libs: []string{"libc.so", "libc.so.96.1"},
			want: []string{"libc.so.96.1"},
		},
		{
			libs: []string{"libc.so.96.1", "libc.so"},
			want: []string{"libc.so.96.1"},
		},
		{
			libs: []string{"libc.a", "libc.so.96.1"},
			want: []string{"libc.a", "libc.so.96.1"},
		},
		{
			libs: []string{"libpthread.so", "libc.so"},
			want: []string{"libc.so", "libpthread.so"},
		},
		{
			libs: []string{"libpthread.so.26.1", "libpthread.so", "libc.so.96.1", "libc.so"},
			want: []string{"libc.so.96.1", "libpthread.so.26.1"},
		},
		{
			libs: []string{"libpthread.so.26.1", "libpthread.so", "libc.so.96.1", "libc.so", "libfoo.so"},
			want: []string{"libc.so.96.1", "libfoo.so", "libpthread.so.26.1"},
		},
	}

	for _, test := range tests {
		t.Run("dedup", func(t *testing.T) {
			got := dedupLibraries(ctxt, test.libs)
			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("dedupLibraries(%v) = %v, want %v", test.libs, got, test.want)
			}
		})
	}
}

func TestDedupLibrariesOpenBSDLink(t *testing.T) {
	// The behavior we're checking for is of interest only on OpenBSD.
	if runtime.GOOS != "openbsd" {
		t.Skip("test only useful on openbsd")
	}

	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	t.Parallel()

	dir := t.TempDir()

	// cgo_import_dynamic both the unversioned libraries and pull in the
	// net package to get a cgo package with a versioned library.
	srcFile := filepath.Join(dir, "x.go")
	src := `package main

import (
	_ "net"
)

//go:cgo_import_dynamic _ _ "libc.so"

func main() {}`
	if err := os.WriteFile(srcFile, []byte(src), 0644); err != nil {
		t.Fatal(err)
	}

	exe := filepath.Join(dir, "deduped.exe")
	out, err := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", exe, srcFile).CombinedOutput()
	if err != nil {
		t.Fatalf("build failure: %s\n%s\n", err, string(out))
	}

	// Result should be runnable.
	if _, err = testenv.Command(t, exe).CombinedOutput(); err != nil {
		t.Fatal(err)
	}
}

func TestLdobjRejectsGoObjectAndUnknownSyso(t *testing.T) {
	dir := t.TempDir()

	// 1. A .syso member disguised as a Go object containing $$ // cgo JSON directives.
	goSyso := filepath.Join(dir, "inject.syso")
	payload := wantHdr + "\n\n$$  // cgo\n[[\"cgo_ldflag\", \"-Wl,-plugin,/tmp/pwn.so\"]]\n$$\n!\n"
	if err := os.WriteFile(goSyso, []byte(payload), 0644); err != nil {
		t.Fatal(err)
	}
	f, err := bio.Open(goSyso)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	oldErrors := nerrors
	oldUnknown := unknownObjFormat
	defer func() {
		nerrors = oldErrors
		unknownObjFormat = oldUnknown
	}()

	er := loader.ErrorReporter{}
	ctxt := &Link{
		loader: loader.NewLoader(0, &er),
	}
	lib := &sym.Library{Pkg: "main"}

	// Using *flagF = true avoids header mismatch or Preload EOF if it proceeds.
	oldFlagF := *flagF
	*flagF = true
	defer func() { *flagF = oldFlagF }()

	// Wait: if ldobj proceeds into Preload on an empty Go object tail, Preload panics or errors,
	// so let's check ctxt.cgodata (populated by ldpkg right before Preload).
	func() {
		defer func() { recover() }()
		ldobj(ctxt, f, lib, int64(len(payload)), "pkg.a(inject.syso)", "pkg.a")
	}()
	if len(ctxt.cgodata) != 0 {
		t.Errorf("ldobj loaded cgo directives from .syso disguised as Go object: %+v", ctxt.cgodata)
	}

	// 2. A .syso member with an unrecognized format (e.g. LLVM bitcode or linker script).
	unknownSyso := filepath.Join(dir, "unknown.syso")
	unknownPayload := "BC\xc0\xde20140000"
	if err := os.WriteFile(unknownSyso, []byte(unknownPayload), 0644); err != nil {
		t.Fatal(err)
	}
	f2, err := bio.Open(unknownSyso)
	if err != nil {
		t.Fatal(err)
	}
	defer f2.Close()

	unknownObjFormat = false
	hobj := ldobj(ctxt, f2, lib, int64(len(unknownPayload)), "pkg.a(unknown.syso)", "pkg.a")
	if unknownObjFormat || hobj != nil {
		t.Errorf("ldobj accepted unrecognized .syso format (unknownObjFormat=%v, hobj=%v)", unknownObjFormat, hobj)
	}
}
