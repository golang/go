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
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
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

	// Notes:
	// - for linux/amd64 and linux/arm64, for relro we'll always see a
	//   .got section when building with -buildmode=pie (in addition
	//   to .dynamic); for some other less mainstream archs (ppc64le,
	//   s390) this is not the case (on ppc64le for example we only
	//   see got refs from C objects). Hence we put ".dynamic" in the
	//   'want RO' list below and ".got" in the 'want RO if present".
	// - when using the external linker, checking for read-only ".got"
	//   is problematic since some linkers will only make the .got
	//   read-only if its size is above a specific threshold, e.g.
	//   https://sourceware.org/git/?p=binutils-gdb.git;a=blob;f=ld/scripttempl/elf.sc;h=d5022fa502f24db23f396f337a6c8978fbc8415b;hb=6fde04116b4b835fa9ec3b3497fcac4e4a0637e2#l74 . For this reason, don't try to verify read-only .got
	//   in the external linking case.

	tests := []struct {
		name                 string
		args                 []string
		prog                 string
		wantSecsRO           []string
		wantSecsROIfPresent  []string
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
			wantSecsRO:           []string{".dynamic"},
			wantSecsROIfPresent:  []string{".got"},
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
			wantSecsRO:           []string{".dynamic"},
			wantSecsROIfPresent:  []string{".got", ".got.plt"},
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
			wantSecsRO:           []string{".dynamic"},
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

			wsrolists := [][]string{test.wantSecsRO, test.wantSecsROIfPresent}
			for k, wsrolist := range wsrolists {
				for _, wsroname := range wsrolist {
					// Locate section of interest.
					var wsro *elf.Section
					for _, s := range elfFile.Sections {
						if s.Name == wsroname {
							wsro = s
							break
						}
					}
					if wsro == nil {
						if k == 0 {
							t.Fatalf("test %s: can't locate %q section",
								test.name, wsroname)
						}
						continue
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

// This program is intended to be just big/complicated enough that
// we wind up with decent-sized .data.rel.ro.{typelink,itablink,gopclntab}
// sections.
const ifacecallsProg = `
package main

import "reflect"

type A string
type B int
type C float64

type describer interface{ What() string }
type timer interface{ When() int }
type rationale interface{ Why() error }

func (a *A) What() string { return "string" }
func (b *B) What() string { return "int" }
func (b *B) When() int    { return int(*b) }
func (b *B) Why() error   { return nil }
func (c *C) What() string { return "float64" }

func i_am_dead(c C) {
	var d describer = &c
	println(d.What())
}

func example(a A, b B) describer {
	if b == 1 {
		return &a
	}
	return &b
}

func ouch(a any, what string) string {
	cv := reflect.ValueOf(a).MethodByName(what).Call(nil)
	return cv[0].String()
}

func main() {
	println(example("", 1).What())
	println(ouch(example("", 1), "What"))
}

`

func TestRelroSectionOverlapIssue67261(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveBuildMode(t, "pie")
	testenv.MustInternalLinkPIE(t)

	// This test case inspired by issue 67261, in which the linker
	// produces a set of sections for -buildmode=pie that confuse the
	// "strip" command, due to overlapping extents. The test first
	// verifies that we don't have any overlapping PROGBITS/DYNAMIC
	// sections, then runs "strip" on the resulting binary.

	dir := t.TempDir()
	src := filepath.Join(dir, "e.go")
	binFile := filepath.Join(dir, "e.exe")

	if err := os.WriteFile(src, []byte(ifacecallsProg), 0666); err != nil {
		t.Fatal(err)
	}

	cmdArgs := []string{"build", "-o", binFile, "-buildmode=pie", "-ldflags=linkmode=internal", src}
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

	// List of interesting sections. Here "interesting" means progbits/dynamic
	// and loadable (has an address), nonzero size.
	secs := []*elf.Section{}
	for _, s := range elfFile.Sections {
		if s.Type != elf.SHT_PROGBITS && s.Type != elf.SHT_DYNAMIC {
			continue
		}
		if s.Addr == 0 || s.Size == 0 {
			continue
		}
		secs = append(secs, s)
	}

	secOverlaps := func(s1, s2 *elf.Section) bool {
		st1 := s1.Addr
		st2 := s2.Addr
		en1 := s1.Addr + s1.Size
		en2 := s2.Addr + s2.Size
		return max(st1, st2) < min(en1, en2)
	}

	// Sort by address
	sort.SliceStable(secs, func(i, j int) bool {
		return secs[i].Addr < secs[j].Addr
	})

	// Check to make sure we don't have any overlaps.
	foundOverlap := false
	for i := 0; i < len(secs)-1; i++ {
		for j := i + 1; j < len(secs); j++ {
			s := secs[i]
			sn := secs[j]
			if secOverlaps(s, sn) {
				t.Errorf("unexpected: section %d:%q (addr=%x size=%x) overlaps section %d:%q (addr=%x size=%x)", i, s.Name, s.Addr, s.Size, i+1, sn.Name, sn.Addr, sn.Size)
				foundOverlap = true
			}
		}
	}
	if foundOverlap {
		// Print some additional info for human inspection.
		t.Logf("** section list follows\n")
		for i := range secs {
			s := secs[i]
			fmt.Printf(" | %2d: ad=0x%08x en=0x%08x sz=0x%08x t=%s %q\n",
				i, s.Addr, s.Addr+s.Size, s.Size, s.Type, s.Name)
		}
	}

	// We need CGO / c-compiler for the next bit.
	testenv.MustHaveCGO(t)

	// Make sure that the resulting binary can be put through strip.
	// Try both "strip" and "llvm-strip"; in each case ask out CC
	// command where to find the tool with "-print-prog-name" (meaning
	// that if CC is gcc, we typically won't be able to find llvm-strip).
	//
	// Interestingly, binutils version of strip will (unfortunately)
	// print error messages if there is a problem but will not return
	// a non-zero exit status (?why?), so we consider any output a
	// failure here.
	stripExecs := []string{}
	ecmd := testenv.Command(t, testenv.GoToolPath(t), "env", "CC")
	if out, err := ecmd.CombinedOutput(); err != nil {
		t.Fatalf("go env CC failed: %v:\n%s", err, out)
	} else {
		ccprog := strings.TrimSpace(string(out))
		tries := []string{"strip", "llvm-strip"}
		for _, try := range tries {
			cmd := testenv.Command(t, ccprog, "-print-prog-name="+try)
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("print-prog-name failed: %+v %v:\n%s",
					cmd.Args, err, out)
			} else {
				sprog := strings.TrimSpace(string(out))
				stripExecs = append(stripExecs, sprog)
			}
		}
	}

	// Run strip on our Go PIE binary, making sure that the strip
	// succeeds and we get no output from strip, then run the resulting
	// stripped binary.
	for k, sprog := range stripExecs {
		if _, err := os.Stat(sprog); err != nil {
			sp1, err := exec.LookPath(sprog)
			if err != nil || sp1 == "" {
				continue
			}
			sprog = sp1
		}
		targ := fmt.Sprintf("p%d.exe", k)
		scmd := testenv.Command(t, sprog, "-o", targ, binFile)
		scmd.Dir = dir
		if sout, serr := scmd.CombinedOutput(); serr != nil {
			t.Fatalf("failed to strip %v: %v:\n%s", scmd.Args, serr, sout)
		} else {
			// Non-empty output indicates failure, as mentioned above.
			if len(string(sout)) != 0 {
				t.Errorf("unexpected outut from %s:\n%s\n", sprog, string(sout))
			}
		}
		rcmd := testenv.Command(t, filepath.Join(dir, targ))
		if out, err := rcmd.CombinedOutput(); err != nil {
			t.Errorf("binary stripped by %s failed: %v:\n%s",
				scmd.Args, err, string(out))
		}
	}

}
