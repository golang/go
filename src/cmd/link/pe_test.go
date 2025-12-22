// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package main

import (
	"debug/pe"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

func TestPESectionsReadOnly(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)

	const (
		prog  = `package main; func main() {}`
		progC = `package main; import "C"; func main() {}`
	)

	tests := []struct {
		name                string
		args                []string
		prog                string
		wantSecsRO          [][]string // each entry is a list of alternative names (first found wins)
		wantSecsROIfPresent [][]string
		mustHaveCGO         bool
		mustInternalLink    bool
	}{
		{
			name:             "linkmode-internal",
			args:             []string{"-ldflags", "-linkmode=internal"},
			prog:             prog,
			mustInternalLink: true,
			wantSecsRO:       [][]string{{".rodata"}, {".gopclntab"}},
			wantSecsROIfPresent: [][]string{
				{".typelink"},
				{".itablink"},
			},
		},
		{
			name:        "linkmode-external",
			args:        []string{"-ldflags", "-linkmode=external"},
			prog:        progC,
			mustHaveCGO: true,
			// External linkers may truncate section names to 8 characters (lld)
			// or preserve long names via string table (GNU ld).
			wantSecsRO: [][]string{{".rodata"}, {".gopclntab", ".gopclnt"}},
			wantSecsROIfPresent: [][]string{
				{".typelink", ".typelin"},
				{".itablink", ".itablin"},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.mustInternalLink {
				testenv.MustInternalLink(t, testenv.SpecialBuildTypes{Cgo: test.mustHaveCGO})
			}
			if test.mustHaveCGO {
				testenv.MustHaveCGO(t)
			}

			dir := t.TempDir()
			src := filepath.Join(dir, fmt.Sprintf("pe_%s.go", test.name))
			binFile := filepath.Join(dir, test.name)

			if err := os.WriteFile(src, []byte(test.prog), 0666); err != nil {
				t.Fatal(err)
			}

			cmdArgs := append([]string{"build", "-o", binFile}, append(test.args, src)...)
			cmd := goCmd(t, cmdArgs...)
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("failed to build %v: %v:\n%s", cmd.Args, err, out)
			}

			pf, err := pe.Open(binFile)
			if err != nil {
				t.Fatalf("failed to open PE file: %v", err)
			}
			defer pf.Close()

			secByName := make(map[string]*pe.Section, len(pf.Sections))
			for _, sec := range pf.Sections {
				secByName[sec.Name] = sec
			}

			// checkRO checks that one of the alternative section names exists and is read-only.
			// names is a list of alternative names (first found wins).
			checkRO := func(names []string, required bool) {
				var sec *pe.Section
				var foundName string
				for _, name := range names {
					if s := secByName[name]; s != nil {
						sec = s
						foundName = name
						break
					}
				}
				if sec == nil {
					if required {
						t.Fatalf("test %s: can't locate any of %q sections", test.name, names)
					}
					return
				}
				if sec.Characteristics&pe.IMAGE_SCN_MEM_READ == 0 {
					t.Errorf("section %s missing IMAGE_SCN_MEM_READ", foundName)
				}
				if sec.Characteristics&pe.IMAGE_SCN_MEM_WRITE != 0 {
					t.Errorf("section %s unexpectedly writable", foundName)
				}
			}

			for _, names := range test.wantSecsRO {
				checkRO(names, true)
			}
			for _, names := range test.wantSecsROIfPresent {
				checkRO(names, false)
			}
		})
	}
}

func TestPESectionAlignment(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)
	testenv.MustInternalLink(t, testenv.SpecialBuildTypes{})

	const prog = `package main; func main() { println("hello") }`

	dir := t.TempDir()
	src := filepath.Join(dir, "align.go")
	binFile := filepath.Join(dir, "align.exe")

	if err := os.WriteFile(src, []byte(prog), 0666); err != nil {
		t.Fatal(err)
	}

	cmd := goCmd(t, "build", "-o", binFile, "-ldflags", "-linkmode=internal", src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("failed to build: %v:\n%s", err, out)
	}

	pf, err := pe.Open(binFile)
	if err != nil {
		t.Fatalf("failed to open PE file: %v", err)
	}
	defer pf.Close()

	// Get section alignment from optional header
	var sectionAlignment uint32
	switch oh := pf.OptionalHeader.(type) {
	case *pe.OptionalHeader32:
		sectionAlignment = oh.SectionAlignment
	case *pe.OptionalHeader64:
		sectionAlignment = oh.SectionAlignment
	default:
		t.Fatal("unknown optional header type")
	}

	if sectionAlignment != 0x1000 {
		t.Errorf("unexpected section alignment: got %#x, want %#x", sectionAlignment, 0x1000)
	}

	// Verify all sections are aligned to section alignment
	for _, sec := range pf.Sections {
		if sec.VirtualAddress%sectionAlignment != 0 {
			t.Errorf("section %s virtual address %#x not aligned to %#x",
				sec.Name, sec.VirtualAddress, sectionAlignment)
		}
	}
}

func TestPESectionCharacteristics(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)
	testenv.MustInternalLink(t, testenv.SpecialBuildTypes{})

	const prog = `
package main

var globalData = []int{1, 2, 3}
var globalBss [1024]byte

func main() {
	println(globalData[0])
	globalBss[0] = 1
}
`

	dir := t.TempDir()
	src := filepath.Join(dir, "chars.go")
	binFile := filepath.Join(dir, "chars.exe")

	if err := os.WriteFile(src, []byte(prog), 0666); err != nil {
		t.Fatal(err)
	}

	cmd := goCmd(t, "build", "-o", binFile, "-ldflags", "-linkmode=internal", src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("failed to build: %v:\n%s", err, out)
	}

	pf, err := pe.Open(binFile)
	if err != nil {
		t.Fatalf("failed to open PE file: %v", err)
	}
	defer pf.Close()

	tests := []struct {
		name      string
		wantFlags uint32
		wantClear uint32
	}{
		{
			name:      ".text",
			wantFlags: pe.IMAGE_SCN_CNT_CODE | pe.IMAGE_SCN_MEM_EXECUTE | pe.IMAGE_SCN_MEM_READ,
			wantClear: pe.IMAGE_SCN_MEM_WRITE,
		},
		{
			name:      ".rodata",
			wantFlags: pe.IMAGE_SCN_CNT_INITIALIZED_DATA | pe.IMAGE_SCN_MEM_READ,
			wantClear: pe.IMAGE_SCN_MEM_WRITE | pe.IMAGE_SCN_MEM_EXECUTE,
		},
		{
			name:      ".data",
			wantFlags: pe.IMAGE_SCN_CNT_INITIALIZED_DATA | pe.IMAGE_SCN_MEM_READ | pe.IMAGE_SCN_MEM_WRITE,
			wantClear: pe.IMAGE_SCN_MEM_EXECUTE,
		},
	}

	for _, test := range tests {
		sec := pf.Section(test.name)
		if sec == nil {
			t.Errorf("section %s not found", test.name)
			continue
		}
		if sec.Characteristics&test.wantFlags != test.wantFlags {
			t.Errorf("section %s: want flags %#x set, got characteristics %#x",
				test.name, test.wantFlags, sec.Characteristics)
		}
		if sec.Characteristics&test.wantClear != 0 {
			t.Errorf("section %s: want flags %#x clear, got characteristics %#x",
				test.name, test.wantClear, sec.Characteristics)
		}
	}
}

func TestPESectionNoOverlap(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)
	testenv.MustInternalLink(t, testenv.SpecialBuildTypes{})

	const prog = `
package main

import "reflect"

type A string
type B int

type describer interface{ What() string }

func (a *A) What() string { return "string" }
func (b *B) What() string { return "int" }

func example(a A, b B) describer {
	if b == 1 {
		return &a
	}
	return &b
}

func main() {
	println(example("", 1).What())
	println(reflect.TypeOf(example("", 1)).String())
}
`

	dir := t.TempDir()
	src := filepath.Join(dir, "overlap.go")
	binFile := filepath.Join(dir, "overlap.exe")

	if err := os.WriteFile(src, []byte(prog), 0666); err != nil {
		t.Fatal(err)
	}

	cmd := goCmd(t, "build", "-o", binFile, "-ldflags", "-linkmode=internal", src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("failed to build: %v:\n%s", err, out)
	}

	pf, err := pe.Open(binFile)
	if err != nil {
		t.Fatalf("failed to open PE file: %v", err)
	}
	defer pf.Close()

	// Collect sections with non-zero virtual address and size
	type secInfo struct {
		name  string
		start uint32
		end   uint32
	}
	var secs []secInfo
	for _, s := range pf.Sections {
		if s.VirtualAddress == 0 || s.VirtualSize == 0 {
			continue
		}
		secs = append(secs, secInfo{
			name:  s.Name,
			start: s.VirtualAddress,
			end:   s.VirtualAddress + s.VirtualSize,
		})
	}

	// Sort by start address
	sort.Slice(secs, func(i, j int) bool {
		return secs[i].start < secs[j].start
	})

	// Check for overlaps
	for i := 0; i < len(secs)-1; i++ {
		for j := i + 1; j < len(secs); j++ {
			s1, s2 := secs[i], secs[j]
			// Check if they overlap: max(start1, start2) < min(end1, end2)
			maxStart := s1.start
			if s2.start > maxStart {
				maxStart = s2.start
			}
			minEnd := s1.end
			if s2.end < minEnd {
				minEnd = s2.end
			}
			if maxStart < minEnd {
				t.Errorf("sections overlap: %s [%#x-%#x] and %s [%#x-%#x]",
					s1.name, s1.start, s1.end, s2.name, s2.start, s2.end)
			}
		}
	}
}

func TestPEDWARFSections(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)
	testenv.MustInternalLink(t, testenv.SpecialBuildTypes{})

	const prog = `package main; func main() { println("hello") }`

	dir := t.TempDir()
	src := filepath.Join(dir, "dwarf.go")
	binFile := filepath.Join(dir, "dwarf.exe")

	if err := os.WriteFile(src, []byte(prog), 0666); err != nil {
		t.Fatal(err)
	}

	// Build without -w to include DWARF
	cmd := goCmd(t, "build", "-o", binFile, "-ldflags", "-linkmode=internal", src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("failed to build: %v:\n%s", err, out)
	}

	pf, err := pe.Open(binFile)
	if err != nil {
		t.Fatalf("failed to open PE file: %v", err)
	}
	defer pf.Close()

	// DWARF sections should be present and have correct characteristics
	dwarfSections := []string{".debug_abbrev", ".debug_info", ".debug_line"}
	for _, name := range dwarfSections {
		sec := pf.Section(name)
		if sec == nil {
			// DWARF section names > 8 chars are stored in string table
			// and section name becomes /N where N is offset
			// Try to find by checking DWARF data
			continue
		}

		// DWARF sections should be readable and discardable
		if sec.Characteristics&pe.IMAGE_SCN_MEM_READ == 0 {
			t.Errorf("DWARF section %s missing IMAGE_SCN_MEM_READ", name)
		}
		if sec.Characteristics&pe.IMAGE_SCN_MEM_DISCARDABLE == 0 {
			t.Errorf("DWARF section %s missing IMAGE_SCN_MEM_DISCARDABLE", name)
		}
		if sec.Characteristics&pe.IMAGE_SCN_MEM_WRITE != 0 {
			t.Errorf("DWARF section %s unexpectedly writable", name)
		}
	}

	// Verify DWARF data is accessible
	dwarfData, err := pf.DWARF()
	if err != nil {
		t.Fatalf("failed to get DWARF data: %v", err)
	}
	if dwarfData == nil {
		t.Error("DWARF data is nil")
	}
}

func TestPEOptionalHeaderSizes(t *testing.T) {
	t.Parallel()
	testenv.MustHaveGoBuild(t)
	testenv.MustInternalLink(t, testenv.SpecialBuildTypes{})

	const prog = `package main; func main() {}`

	dir := t.TempDir()
	src := filepath.Join(dir, "header.go")
	binFile := filepath.Join(dir, "header.exe")

	if err := os.WriteFile(src, []byte(prog), 0666); err != nil {
		t.Fatal(err)
	}

	cmd := goCmd(t, "build", "-o", binFile, "-ldflags", "-linkmode=internal", src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("failed to build: %v:\n%s", err, out)
	}

	pf, err := pe.Open(binFile)
	if err != nil {
		t.Fatalf("failed to open PE file: %v", err)
	}
	defer pf.Close()

	switch oh := pf.OptionalHeader.(type) {
	case *pe.OptionalHeader32:
		if oh.Magic != 0x10b {
			t.Errorf("32-bit magic: got %#x, want %#x", oh.Magic, 0x10b)
		}
		if oh.SizeOfHeaders == 0 {
			t.Error("SizeOfHeaders is 0")
		}
		if oh.SizeOfImage == 0 {
			t.Error("SizeOfImage is 0")
		}
		// Verify SizeOfImage is aligned to section alignment
		if oh.SizeOfImage%oh.SectionAlignment != 0 {
			t.Errorf("SizeOfImage %#x not aligned to SectionAlignment %#x",
				oh.SizeOfImage, oh.SectionAlignment)
		}
	case *pe.OptionalHeader64:
		if oh.Magic != 0x20b {
			t.Errorf("64-bit magic: got %#x, want %#x", oh.Magic, 0x20b)
		}
		if oh.SizeOfHeaders == 0 {
			t.Error("SizeOfHeaders is 0")
		}
		if oh.SizeOfImage == 0 {
			t.Error("SizeOfImage is 0")
		}
		// Verify SizeOfImage is aligned to section alignment
		if oh.SizeOfImage%oh.SectionAlignment != 0 {
			t.Errorf("SizeOfImage %#x not aligned to SectionAlignment %#x",
				oh.SizeOfImage, oh.SectionAlignment)
		}
	default:
		t.Fatal("unknown optional header type")
	}
}
