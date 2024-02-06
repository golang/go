// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package ld

import (
	"debug/elf"
	"fmt"
	"internal/testenv"
	"os"
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
	if err := os.WriteFile(src, []byte(prog), 0666); err != nil {
		t.Fatal(err)
	}

	binFile := filepath.Join(dir, "issue33358")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", binFile, src)
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
	case "linux-amd64", "linux-arm64", "freebsd-amd64", "openbsd-amd64":
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
	out, err := testenv.Command(t, testenv.GoToolPath(t), argv...).CombinedOutput()
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

func TestShStrTabAttributesIssue62600(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)
	dir := t.TempDir()

	const prog = `
package main

func main() {
	println("whee")
}
`
	src := filepath.Join(dir, "issue62600.go")
	if err := os.WriteFile(src, []byte(prog), 0666); err != nil {
		t.Fatal(err)
	}

	binFile := filepath.Join(dir, "issue62600")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", binFile, src)
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

	section := elfFile.Section(".shstrtab")
	if section == nil {
		t.Fatal("no .shstrtab")
	}

	// The .shstrtab section should have a zero address, non-zero
	// size, no ALLOC flag, and the offset should not fall into any of
	// the segments defined by the program headers.
	if section.Addr != 0 {
		t.Fatalf("expected Addr == 0 for .shstrtab got %x", section.Addr)
	}
	if section.Size == 0 {
		t.Fatal("expected nonzero Size for .shstrtab got 0")
	}
	if section.Flags&elf.SHF_ALLOC != 0 {
		t.Fatal("expected zero alloc flag got nonzero for .shstrtab")
	}
	for idx, p := range elfFile.Progs {
		if section.Offset >= p.Off && section.Offset < p.Off+p.Filesz {
			t.Fatalf("badly formed .shstrtab, is contained in segment %d", idx)
		}
	}
}

func TestElfBindNow(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)

	const (
		prog  = `package main; func main() {}`
		progC = `package main; import "C"; func main() {}`
	)

	tests := []struct {
		name                 string
		args                 []string
		prog                 string
		mustHaveCGO          bool
		mustHaveBuildModePIE bool
		wantDf1Now           bool
		wantDf1Pie           bool
		wantDfBindNow        bool
	}{
		{name: "default", prog: prog},
		{
			name:                 "pie",
			args:                 []string{"-buildmode=pie", "-ldflags", "-linkmode=internal"},
			mustHaveBuildModePIE: true,
			prog:                 prog,
			wantDf1Pie:           true,
		},
		{
			name:          "bindnow",
			args:          []string{"-ldflags", "-bindnow -linkmode=internal"},
			prog:          progC,
			mustHaveCGO:   true,
			wantDf1Now:    true,
			wantDfBindNow: true,
		},
		{
			name:                 "bindnow-pie",
			args:                 []string{"-buildmode=pie", "-ldflags", "-bindnow -linkmode=internal"},
			prog:                 prog,
			mustHaveBuildModePIE: true,
			wantDf1Now:           true,
			wantDf1Pie:           true,
			wantDfBindNow:        true,
		},
	}

	gotDynFlag := func(flags []uint64, dynFlag uint64) bool {
		for _, flag := range flags {
			if gotFlag := dynFlag&flag != 0; gotFlag {
				return true
			}
		}

		return false
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			testenv.MustInternalLink(t, test.mustHaveCGO)
			if test.mustHaveCGO {
				testenv.MustHaveCGO(t)
			}
			if test.mustHaveBuildModePIE {
				testenv.MustHaveBuildMode(t, "pie")
			}

			var (
				gotDfBindNow, gotDf1Now, gotDf1Pie bool

				dir     = t.TempDir()
				src     = filepath.Join(dir, fmt.Sprintf("elf_%s.go", test.name))
				binFile = filepath.Join(dir, test.name)
			)

			if err := os.WriteFile(src, []byte(test.prog), 0666); err != nil {
				t.Fatal(err)
			}

			cmdArgs := append([]string{"build", "-o", binFile}, append(test.args, src)...)
			cmd := testenv.Command(t, testenv.GoToolPath(t), cmdArgs...)

			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("failed to build %v: %v:\n%s", cmd.Args, err, out)
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
			defer elfFile.Close()

			flags, err := elfFile.DynValue(elf.DT_FLAGS)
			if err != nil {
				t.Fatalf("failed to get DT_FLAGS: %v", err)
			}

			flags1, err := elfFile.DynValue(elf.DT_FLAGS_1)
			if err != nil {
				t.Fatalf("failed to get DT_FLAGS_1: %v", err)
			}

			if gotDfBindNow = gotDynFlag(flags, uint64(elf.DF_BIND_NOW)); gotDfBindNow != test.wantDfBindNow {
				t.Fatalf("DT_FLAGS BIND_NOW flag is %v, want: %v", gotDfBindNow, test.wantDfBindNow)
			}

			if gotDf1Now = gotDynFlag(flags1, uint64(elf.DF_1_NOW)); gotDf1Now != test.wantDf1Now {
				t.Fatalf("DT_FLAGS_1 DF_1_NOW flag is %v, want: %v", gotDf1Now, test.wantDf1Now)
			}

			if gotDf1Pie = gotDynFlag(flags1, uint64(elf.DF_1_PIE)); gotDf1Pie != test.wantDf1Pie {
				t.Fatalf("DT_FLAGS_1 DF_1_PIE flag is %v, want: %v", gotDf1Pie, test.wantDf1Pie)
			}
		})
	}
}
