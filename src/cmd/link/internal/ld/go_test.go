// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"internal/testenv"
	"io/ioutil"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
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
	if err := ioutil.WriteFile(srcFile, []byte(src), 0644); err != nil {
		t.Fatal(err)
	}

	exe := filepath.Join(dir, "deduped.exe")
	out, err := exec.Command(testenv.GoToolPath(t), "build", "-o", exe, srcFile).CombinedOutput()
	if err != nil {
		t.Fatalf("build failure: %s\n%s\n", err, string(out))
	}

	// Result should be runnable.
	if _, err = exec.Command(exe).CombinedOutput(); err != nil {
		t.Fatal(err)
	}
}
