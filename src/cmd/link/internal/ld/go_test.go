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

func TestSetCgoAttrValidatesExtnameAndDynlinker(t *testing.T) {
	oldErrors := nerrors
	oldInterp := interpreter
	defer func() {
		nerrors = oldErrors
		interpreter = oldInterp
	}()

	er := loader.ErrorReporter{}
	ctxt := &Link{
		loader: loader.NewLoader(0, &er),
	}
	hostObjSyms := make(map[loader.Sym]struct{})

	// 1. Exported symbol must not have its SymExtname overwritten by a subsequent cgo_import_dynamic.
	s := ctxt.loader.LookupOrCreateSym("_cgo_panic", 0)
	su := ctxt.loader.MakeSymbolUpdater(s)
	su.SetType(sym.SBSS)
	setCgoAttr(ctxt, "test.a", "runtime/cgo", [][]string{
		{"cgo_export_dynamic", "_cgo_panic", "_cgo_panic"},
		{"cgo_import_dynamic", "_cgo_panic", "overwritten_panic"},
	}, hostObjSyms)
	if got := ctxt.loader.SymExtname(s); got != "_cgo_panic" {
		t.Errorf("cgo_import_dynamic overwrote exported symbol SymExtname: got %q, want %q", got, "_cgo_panic")
	}

	// Also verify that when cgo_import_dynamic (without a library) precedes
	// cgo_export_dynamic, the export overrides the import and clears SDYNIMPORT.
	s2 := ctxt.loader.LookupOrCreateSym("_cgo_export_after_import", 0)
	setCgoAttr(ctxt, "test.a", "runtime/cgo", [][]string{
		{"cgo_import_dynamic", "_cgo_export_after_import", "imported_name"},
		{"cgo_export_dynamic", "_cgo_export_after_import", "_cgo_export_after_import"},
	}, hostObjSyms)
	if got := ctxt.loader.SymExtname(s2); got != "_cgo_export_after_import" {
		t.Errorf("cgo_export_dynamic did not override prior cgo_import_dynamic SymExtname: got %q, want %q", got, "_cgo_export_after_import")
	}
	if got := ctxt.loader.SymType(s2); got == sym.SDYNIMPORT {
		t.Errorf("cgo_export_dynamic did not clear SDYNIMPORT from prior cgo_import_dynamic")
	}

	// 2. Comma injection in cgo_import_dynamic / cgo_export_dynamic / cgo_dynamic_linker must be rejected.
	nerrors = 0
	interpreter = ""
	setCgoAttr(ctxt, "test.a", "main", [][]string{
		{"cgo_import_dynamic", "foo", "foo,--plugin=/tmp/pwn.so"},
	}, hostObjSyms)
	if nerrors == 0 {
		t.Errorf("expected error for comma in cgo_import_dynamic remote symbol, got nerrors=0")
	}

	nerrors = 0
	setCgoAttr(ctxt, "test.a", "main", [][]string{
		{"cgo_export_dynamic", "bar", "bar,--plugin=/tmp/pwn.so"},
	}, hostObjSyms)
	if nerrors == 0 {
		t.Errorf("expected error for comma in cgo_export_dynamic remote symbol, got nerrors=0")
	}

	nerrors = 0
	setCgoAttr(ctxt, "test.a", "main", [][]string{
		{"cgo_dynamic_linker", "/lib64/ld.so,--plugin=/tmp/pwn.so"},
	}, hostObjSyms)
	if nerrors == 0 || interpreter != "" {
		t.Errorf("expected error for comma in cgo_dynamic_linker, got nerrors=%d interpreter=%q", nerrors, interpreter)
	}
}

func TestPEImportedDLLsValidation(t *testing.T) {
	oldDr := dr
	oldErrors := nerrors
	defer func() {
		dr = oldDr
		nerrors = oldErrors
	}()

	dr = &Dll{
		name: "../../../../../../tmp/evil.dll",
		next: &Dll{
			name: "kernel32.dll",
		},
	}
	nerrors = 0
	dlls := peimporteddlls()
	if nerrors == 0 {
		t.Errorf("expected error for path traversal in PE imported DLL name, got nerrors=0 (dlls=%v)", dlls)
	}
	for _, arg := range dlls {
		if arg == "-l../../../../../../tmp/evil" {
			t.Errorf("peimporteddlls emitted unsafe -l path traversal flag: %q", arg)
		}
	}
}
