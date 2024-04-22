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
		prog = `package main; func main() {}`
		// with default buildmode code compiles in a statically linked binary, hence CGO
		progC = `package main; import "C"; func main() {}`
	)

	tests := []struct {
		name                 string
		args                 []string
		prog                 string
		wantSecsRO           []string
		mustHaveBuildModePIE bool
		mustHaveCGO          bool
		mustInternalLink     bool
		wantDfBindNow        bool
		wantDf1Now           bool
		wantDf1Pie           bool
	}{
		{name: "default", prog: prog},
		{
			name:                 "pie-linkmode-internal",
			args:                 []string{"-buildmode=pie", "-ldflags", "-linkmode=internal"},
			prog:                 prog,
			mustHaveBuildModePIE: true,
			mustInternalLink:     true,
			wantDf1Pie:           true,
			wantSecsRO:           []string{".dynamic", ".got"},
		},
		{
			name:             "bindnow-linkmode-internal",
			args:             []string{"-ldflags", "-bindnow -linkmode=internal"},
			prog:             progC,
			mustHaveCGO:      true,
			mustInternalLink: true,
			wantDfBindNow:    true,
			wantDf1Now:       true,
		},
		{
			name:                 "bindnow-pie-linkmode-internal",
			args:                 []string{"-buildmode=pie", "-ldflags", "-bindnow -linkmode=internal"},
			prog:                 prog,
			mustHaveBuildModePIE: true,
			mustInternalLink:     true,
			wantDfBindNow:        true,
			wantDf1Now:           true,
			wantDf1Pie:           true,
			wantSecsRO:           []string{".dynamic", ".got", ".got.plt"},
		},
		{
			name:                 "bindnow-pie-linkmode-external",
			args:                 []string{"-buildmode=pie", "-ldflags", "-bindnow -linkmode=external"},
			prog:                 prog,
			mustHaveBuildModePIE: true,
			mustHaveCGO:          true,
			wantDfBindNow:        true,
			wantDf1Now:           true,
			wantDf1Pie:           true,
			// NB: external linker produces .plt.got, not .got.plt
			wantSecsRO: []string{".dynamic", ".got"},
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

	segContainsSec := func(p *elf.Prog, s *elf.Section) bool {
		return s.Addr >= p.Vaddr &&
			s.Addr+s.FileSize <= p.Vaddr+p.Filesz
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.mustInternalLink {
				testenv.MustInternalLink(t, test.mustHaveCGO)
			}
			if test.mustHaveCGO {
				testenv.MustHaveCGO(t)
			}
			if test.mustHaveBuildModePIE {
				testenv.MustHaveBuildMode(t, "pie")
			}
			if test.mustHaveBuildModePIE && test.mustInternalLink {
				testenv.MustInternalLinkPIE(t)
			}

			var (
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

			gotDfBindNow := gotDynFlag(flags, uint64(elf.DF_BIND_NOW))
			gotDf1Now := gotDynFlag(flags1, uint64(elf.DF_1_NOW))

			bindNowFlagsMatch := gotDfBindNow == test.wantDfBindNow && gotDf1Now == test.wantDf1Now

			// some external linkers may set one of the two flags but not both.
			if !test.mustInternalLink {
				bindNowFlagsMatch = gotDfBindNow == test.wantDfBindNow || gotDf1Now == test.wantDf1Now
			}

			if !bindNowFlagsMatch {
				t.Fatalf("Dynamic flags mismatch:\n"+
					"DT_FLAGS BIND_NOW	got: %v,	want: %v\n"+
					"DT_FLAGS_1 DF_1_NOW	got: %v,	want: %v",
					gotDfBindNow, test.wantDfBindNow, gotDf1Now, test.wantDf1Now)
			}

			if gotDf1Pie := gotDynFlag(flags1, uint64(elf.DF_1_PIE)); gotDf1Pie != test.wantDf1Pie {
				t.Fatalf("DT_FLAGS_1 DF_1_PIE got: %v, want: %v", gotDf1Pie, test.wantDf1Pie)
			}

			// Skipping this newer portion of the test temporarily pending resolution of problems on ppc64le, loonpg64, possibly others.
			if false {

				for _, wsroname := range test.wantSecsRO {
					// Locate section of interest.
					var wsro *elf.Section
					for _, s := range elfFile.Sections {
						if s.Name == wsroname {
							wsro = s
							break
						}
					}
					if wsro == nil {
						t.Fatalf("test %s: can't locate %q section",
							test.name, wsroname)
					}

					// Now walk the program headers. Section should be part of
					// some segment that is readonly.
					foundRO := false
					foundSegs := []*elf.Prog{}
					for _, p := range elfFile.Progs {
						if segContainsSec(p, wsro) {
							foundSegs = append(foundSegs, p)
							if p.Flags == elf.PF_R {
								foundRO = true
							}
						}
					}
					if !foundRO {
						// Things went off the rails. Write out some
						// useful information for a human looking at the
						// test failure.
						t.Logf("test %s: %q section not in readonly segment",
							wsro.Name, test.name)
						t.Logf("section %s location: st=0x%x en=0x%x\n",
							wsro.Name, wsro.Addr, wsro.Addr+wsro.FileSize)
						t.Logf("sec %s found in these segments: ", wsro.Name)
						for _, p := range foundSegs {
							t.Logf(" %q", p.Type)
						}
						t.Logf("\nall segments: \n")
						for k, p := range elfFile.Progs {
							t.Logf("%d t=%s fl=%s st=0x%x en=0x%x\n",
								k, p.Type, p.Flags, p.Vaddr, p.Vaddr+p.Filesz)
						}
						t.Fatalf("test %s failed", test.name)
					}
				}
			}
		})
	}
}
