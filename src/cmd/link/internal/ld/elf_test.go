//go:build cgo
// +build cgo

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"debug/elf"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestDynSymShInfo(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()

	const prog = `
package main

import "net"

func main() {
	net.Dial("", "")
}
`
	src := filepath.Join(dir, "issue33358.go")
	if err := ioutil.WriteFile(src, []byte(prog), 0666); err != nil {
		t.Fatal(err)
	}

	binFile := filepath.Join(dir, "issue33358")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", binFile, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%v: %v:\n%s", cmd.Args, err, out)
	}

	fi, err := os.Open(binFile)
	if err != nil {
		t.Fatalf("failed to open built file: %v", err)
	}
	defer fi.Close()

	elfFile, err := elf.NewFile(fi)
	if err != nil {
		t.Skip("The system may not support ELF, skipped.")
	}

	section := elfFile.Section(".dynsym")
	if section == nil {
		t.Fatal("no dynsym")
	}

	symbols, err := elfFile.DynamicSymbols()
	if err != nil {
		t.Fatalf("failed to get dynamic symbols: %v", err)
	}

	var numLocalSymbols uint32
	for i, s := range symbols {
		if elf.ST_BIND(s.Info) != elf.STB_LOCAL {
			numLocalSymbols = uint32(i + 1)
			break
		}
	}

	if section.Info != numLocalSymbols {
		t.Fatalf("Unexpected sh info, want greater than 0, got: %d", section.Info)
	}
}

func TestNoDuplicateNeededEntries(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)

	// run this test on just a small set of platforms (no need to test it
	// across the board given the nature of the test).
	pair := runtime.GOOS + "-" + runtime.GOARCH
	switch pair {
	case "linux-amd64", "freebsd-amd64", "openbsd-amd64":
	default:
		t.Skip("no need for test on " + pair)
	}

	t.Parallel()

	dir := t.TempDir()

	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Failed to get working directory: %v", err)
	}

	path := filepath.Join(dir, "x")
	argv := []string{"build", "-o", path, filepath.Join(wd, "testdata", "issue39256")}
	out, err := exec.Command(testenv.GoToolPath(t), argv...).CombinedOutput()
	if err != nil {
		t.Fatalf("Build failure: %s\n%s\n", err, string(out))
	}

	f, err := elf.Open(path)
	if err != nil {
		t.Fatalf("Failed to open ELF file: %v", err)
	}
	libs, err := f.ImportedLibraries()
	if err != nil {
		t.Fatalf("Failed to read imported libraries: %v", err)
	}

	var count int
	for _, lib := range libs {
		if lib == "libc.so" || strings.HasPrefix(lib, "libc.so.") {
			count++
		}
	}

	if got, want := count, 1; got != want {
		t.Errorf("Got %d entries for `libc.so`, want %d", got, want)
	}
}
