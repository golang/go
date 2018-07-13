// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	"debug/dwarf"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"testing"
	"text/template"
)

type fileTest struct {
	file           string
	hdr            FileHeader
	opthdr         interface{}
	sections       []*SectionHeader
	symbols        []*Symbol
	hasNoDwarfInfo bool
}

var fileTests = []fileTest{
	{
		file: "testdata/gcc-386-mingw-obj",
		hdr:  FileHeader{0x014c, 0x000c, 0x0, 0x64a, 0x1e, 0x0, 0x104},
		sections: []*SectionHeader{
			{".text", 0, 0, 36, 500, 1440, 0, 3, 0, 0x60300020},
			{".data", 0, 0, 0, 0, 0, 0, 0, 0, 3224371264},
			{".bss", 0, 0, 0, 0, 0, 0, 0, 0, 3224371328},
			{".debug_abbrev", 0, 0, 137, 536, 0, 0, 0, 0, 0x42100000},
			{".debug_info", 0, 0, 418, 673, 1470, 0, 7, 0, 1108344832},
			{".debug_line", 0, 0, 128, 1091, 1540, 0, 1, 0, 1108344832},
			{".rdata", 0, 0, 16, 1219, 0, 0, 0, 0, 1076887616},
			{".debug_frame", 0, 0, 52, 1235, 1550, 0, 2, 0, 1110441984},
			{".debug_loc", 0, 0, 56, 1287, 0, 0, 0, 0, 1108344832},
			{".debug_pubnames", 0, 0, 27, 1343, 1570, 0, 1, 0, 1108344832},
			{".debug_pubtypes", 0, 0, 38, 1370, 1580, 0, 1, 0, 1108344832},
			{".debug_aranges", 0, 0, 32, 1408, 1590, 0, 2, 0, 1108344832},
		},
		symbols: []*Symbol{
			{".file", 0x0, -2, 0x0, 0x67},
			{"_main", 0x0, 1, 0x20, 0x2},
			{".text", 0x0, 1, 0x0, 0x3},
			{".data", 0x0, 2, 0x0, 0x3},
			{".bss", 0x0, 3, 0x0, 0x3},
			{".debug_abbrev", 0x0, 4, 0x0, 0x3},
			{".debug_info", 0x0, 5, 0x0, 0x3},
			{".debug_line", 0x0, 6, 0x0, 0x3},
			{".rdata", 0x0, 7, 0x0, 0x3},
			{".debug_frame", 0x0, 8, 0x0, 0x3},
			{".debug_loc", 0x0, 9, 0x0, 0x3},
			{".debug_pubnames", 0x0, 10, 0x0, 0x3},
			{".debug_pubtypes", 0x0, 11, 0x0, 0x3},
			{".debug_aranges", 0x0, 12, 0x0, 0x3},
			{"___main", 0x0, 0, 0x20, 0x2},
			{"_puts", 0x0, 0, 0x20, 0x2},
		},
	},
	{
		file: "testdata/gcc-386-mingw-exec",
		hdr:  FileHeader{0x014c, 0x000f, 0x4c6a1b60, 0x3c00, 0x282, 0xe0, 0x107},
		opthdr: &OptionalHeader32{
			0x10b, 0x2, 0x38, 0xe00, 0x1a00, 0x200, 0x1160, 0x1000, 0x2000, 0x400000, 0x1000, 0x200, 0x4, 0x0, 0x1, 0x0, 0x4, 0x0, 0x0, 0x10000, 0x400, 0x14abb, 0x3, 0x0, 0x200000, 0x1000, 0x100000, 0x1000, 0x0, 0x10,
			[16]DataDirectory{
				{0x0, 0x0},
				{0x5000, 0x3c8},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x7000, 0x18},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
			},
		},
		sections: []*SectionHeader{
			{".text", 0xcd8, 0x1000, 0xe00, 0x400, 0x0, 0x0, 0x0, 0x0, 0x60500060},
			{".data", 0x10, 0x2000, 0x200, 0x1200, 0x0, 0x0, 0x0, 0x0, 0xc0300040},
			{".rdata", 0x120, 0x3000, 0x200, 0x1400, 0x0, 0x0, 0x0, 0x0, 0x40300040},
			{".bss", 0xdc, 0x4000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc0400080},
			{".idata", 0x3c8, 0x5000, 0x400, 0x1600, 0x0, 0x0, 0x0, 0x0, 0xc0300040},
			{".CRT", 0x18, 0x6000, 0x200, 0x1a00, 0x0, 0x0, 0x0, 0x0, 0xc0300040},
			{".tls", 0x20, 0x7000, 0x200, 0x1c00, 0x0, 0x0, 0x0, 0x0, 0xc0300040},
			{".debug_aranges", 0x20, 0x8000, 0x200, 0x1e00, 0x0, 0x0, 0x0, 0x0, 0x42100000},
			{".debug_pubnames", 0x51, 0x9000, 0x200, 0x2000, 0x0, 0x0, 0x0, 0x0, 0x42100000},
			{".debug_pubtypes", 0x91, 0xa000, 0x200, 0x2200, 0x0, 0x0, 0x0, 0x0, 0x42100000},
			{".debug_info", 0xe22, 0xb000, 0x1000, 0x2400, 0x0, 0x0, 0x0, 0x0, 0x42100000},
			{".debug_abbrev", 0x157, 0xc000, 0x200, 0x3400, 0x0, 0x0, 0x0, 0x0, 0x42100000},
			{".debug_line", 0x144, 0xd000, 0x200, 0x3600, 0x0, 0x0, 0x0, 0x0, 0x42100000},
			{".debug_frame", 0x34, 0xe000, 0x200, 0x3800, 0x0, 0x0, 0x0, 0x0, 0x42300000},
			{".debug_loc", 0x38, 0xf000, 0x200, 0x3a00, 0x0, 0x0, 0x0, 0x0, 0x42100000},
		},
	},
	{
		file: "testdata/gcc-386-mingw-no-symbols-exec",
		hdr:  FileHeader{0x14c, 0x8, 0x69676572, 0x0, 0x0, 0xe0, 0x30f},
		opthdr: &OptionalHeader32{0x10b, 0x2, 0x18, 0xe00, 0x1e00, 0x200, 0x1280, 0x1000, 0x2000, 0x400000, 0x1000, 0x200, 0x4, 0x0, 0x1, 0x0, 0x4, 0x0, 0x0, 0x9000, 0x400, 0x5306, 0x3, 0x0, 0x200000, 0x1000, 0x100000, 0x1000, 0x0, 0x10,
			[16]DataDirectory{
				{0x0, 0x0},
				{0x6000, 0x378},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x8004, 0x18},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x60b8, 0x7c},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
			},
		},
		sections: []*SectionHeader{
			{".text", 0xc64, 0x1000, 0xe00, 0x400, 0x0, 0x0, 0x0, 0x0, 0x60500060},
			{".data", 0x10, 0x2000, 0x200, 0x1200, 0x0, 0x0, 0x0, 0x0, 0xc0300040},
			{".rdata", 0x134, 0x3000, 0x200, 0x1400, 0x0, 0x0, 0x0, 0x0, 0x40300040},
			{".eh_fram", 0x3a0, 0x4000, 0x400, 0x1600, 0x0, 0x0, 0x0, 0x0, 0x40300040},
			{".bss", 0x60, 0x5000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc0300080},
			{".idata", 0x378, 0x6000, 0x400, 0x1a00, 0x0, 0x0, 0x0, 0x0, 0xc0300040},
			{".CRT", 0x18, 0x7000, 0x200, 0x1e00, 0x0, 0x0, 0x0, 0x0, 0xc0300040},
			{".tls", 0x20, 0x8000, 0x200, 0x2000, 0x0, 0x0, 0x0, 0x0, 0xc0300040},
		},
		hasNoDwarfInfo: true,
	},
	{
		file: "testdata/gcc-amd64-mingw-obj",
		hdr:  FileHeader{0x8664, 0x6, 0x0, 0x198, 0x12, 0x0, 0x4},
		sections: []*SectionHeader{
			{".text", 0x0, 0x0, 0x30, 0x104, 0x15c, 0x0, 0x3, 0x0, 0x60500020},
			{".data", 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc0500040},
			{".bss", 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc0500080},
			{".rdata", 0x0, 0x0, 0x10, 0x134, 0x0, 0x0, 0x0, 0x0, 0x40500040},
			{".xdata", 0x0, 0x0, 0xc, 0x144, 0x0, 0x0, 0x0, 0x0, 0x40300040},
			{".pdata", 0x0, 0x0, 0xc, 0x150, 0x17a, 0x0, 0x3, 0x0, 0x40300040},
		},
		symbols: []*Symbol{
			{".file", 0x0, -2, 0x0, 0x67},
			{"main", 0x0, 1, 0x20, 0x2},
			{".text", 0x0, 1, 0x0, 0x3},
			{".data", 0x0, 2, 0x0, 0x3},
			{".bss", 0x0, 3, 0x0, 0x3},
			{".rdata", 0x0, 4, 0x0, 0x3},
			{".xdata", 0x0, 5, 0x0, 0x3},
			{".pdata", 0x0, 6, 0x0, 0x3},
			{"__main", 0x0, 0, 0x20, 0x2},
			{"puts", 0x0, 0, 0x20, 0x2},
		},
		hasNoDwarfInfo: true,
	},
	{
		file: "testdata/gcc-amd64-mingw-exec",
		hdr:  FileHeader{0x8664, 0x11, 0x53e4364f, 0x39600, 0x6fc, 0xf0, 0x27},
		opthdr: &OptionalHeader64{
			0x20b, 0x2, 0x16, 0x6a00, 0x2400, 0x1600, 0x14e0, 0x1000, 0x400000, 0x1000, 0x200, 0x4, 0x0, 0x0, 0x0, 0x5, 0x2, 0x0, 0x45000, 0x600, 0x46f19, 0x3, 0x0, 0x200000, 0x1000, 0x100000, 0x1000, 0x0, 0x10,
			[16]DataDirectory{
				{0x0, 0x0},
				{0xe000, 0x990},
				{0x0, 0x0},
				{0xa000, 0x498},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x10000, 0x28},
				{0x0, 0x0},
				{0x0, 0x0},
				{0xe254, 0x218},
				{0x0, 0x0},
				{0x0, 0x0},
				{0x0, 0x0},
			}},
		sections: []*SectionHeader{
			{".text", 0x6860, 0x1000, 0x6a00, 0x600, 0x0, 0x0, 0x0, 0x0, 0x60500020},
			{".data", 0xe0, 0x8000, 0x200, 0x7000, 0x0, 0x0, 0x0, 0x0, 0xc0500040},
			{".rdata", 0x6b0, 0x9000, 0x800, 0x7200, 0x0, 0x0, 0x0, 0x0, 0x40600040},
			{".pdata", 0x498, 0xa000, 0x600, 0x7a00, 0x0, 0x0, 0x0, 0x0, 0x40300040},
			{".xdata", 0x488, 0xb000, 0x600, 0x8000, 0x0, 0x0, 0x0, 0x0, 0x40300040},
			{".bss", 0x1410, 0xc000, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xc0600080},
			{".idata", 0x990, 0xe000, 0xa00, 0x8600, 0x0, 0x0, 0x0, 0x0, 0xc0300040},
			{".CRT", 0x68, 0xf000, 0x200, 0x9000, 0x0, 0x0, 0x0, 0x0, 0xc0400040},
			{".tls", 0x48, 0x10000, 0x200, 0x9200, 0x0, 0x0, 0x0, 0x0, 0xc0600040},
			{".debug_aranges", 0x600, 0x11000, 0x600, 0x9400, 0x0, 0x0, 0x0, 0x0, 0x42500040},
			{".debug_info", 0x1316e, 0x12000, 0x13200, 0x9a00, 0x0, 0x0, 0x0, 0x0, 0x42100040},
			{".debug_abbrev", 0x2ccb, 0x26000, 0x2e00, 0x1cc00, 0x0, 0x0, 0x0, 0x0, 0x42100040},
			{".debug_line", 0x3c4d, 0x29000, 0x3e00, 0x1fa00, 0x0, 0x0, 0x0, 0x0, 0x42100040},
			{".debug_frame", 0x18b8, 0x2d000, 0x1a00, 0x23800, 0x0, 0x0, 0x0, 0x0, 0x42400040},
			{".debug_str", 0x396, 0x2f000, 0x400, 0x25200, 0x0, 0x0, 0x0, 0x0, 0x42100040},
			{".debug_loc", 0x13240, 0x30000, 0x13400, 0x25600, 0x0, 0x0, 0x0, 0x0, 0x42100040},
			{".debug_ranges", 0xa70, 0x44000, 0xc00, 0x38a00, 0x0, 0x0, 0x0, 0x0, 0x42100040},
		},
	},
}

func isOptHdrEq(a, b interface{}) bool {
	switch va := a.(type) {
	case *OptionalHeader32:
		vb, ok := b.(*OptionalHeader32)
		if !ok {
			return false
		}
		return *vb == *va
	case *OptionalHeader64:
		vb, ok := b.(*OptionalHeader64)
		if !ok {
			return false
		}
		return *vb == *va
	case nil:
		return b == nil
	}
	return false
}

func TestOpen(t *testing.T) {
	for i := range fileTests {
		tt := &fileTests[i]

		f, err := Open(tt.file)
		if err != nil {
			t.Error(err)
			continue
		}
		if !reflect.DeepEqual(f.FileHeader, tt.hdr) {
			t.Errorf("open %s:\n\thave %#v\n\twant %#v\n", tt.file, f.FileHeader, tt.hdr)
			continue
		}
		if !isOptHdrEq(tt.opthdr, f.OptionalHeader) {
			t.Errorf("open %s:\n\thave %#v\n\twant %#v\n", tt.file, f.OptionalHeader, tt.opthdr)
			continue
		}

		for i, sh := range f.Sections {
			if i >= len(tt.sections) {
				break
			}
			have := &sh.SectionHeader
			want := tt.sections[i]
			if !reflect.DeepEqual(have, want) {
				t.Errorf("open %s, section %d:\n\thave %#v\n\twant %#v\n", tt.file, i, have, want)
			}
		}
		tn := len(tt.sections)
		fn := len(f.Sections)
		if tn != fn {
			t.Errorf("open %s: len(Sections) = %d, want %d", tt.file, fn, tn)
		}
		for i, have := range f.Symbols {
			if i >= len(tt.symbols) {
				break
			}
			want := tt.symbols[i]
			if !reflect.DeepEqual(have, want) {
				t.Errorf("open %s, symbol %d:\n\thave %#v\n\twant %#v\n", tt.file, i, have, want)
			}
		}
		if !tt.hasNoDwarfInfo {
			_, err = f.DWARF()
			if err != nil {
				t.Errorf("fetching %s dwarf details failed: %v", tt.file, err)
			}
		}
	}
}

func TestOpenFailure(t *testing.T) {
	filename := "file.go"    // not a PE file
	_, err := Open(filename) // don't crash
	if err == nil {
		t.Errorf("open %s: succeeded unexpectedly", filename)
	}
}

const (
	linkNoCgo = iota
	linkCgoDefault
	linkCgoInternal
	linkCgoExternal
)

func testDWARF(t *testing.T, linktype int) {
	if runtime.GOOS != "windows" {
		t.Skip("skipping windows only test")
	}
	testenv.MustHaveGoRun(t)

	tmpdir, err := ioutil.TempDir("", "TestDWARF")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	src := filepath.Join(tmpdir, "a.go")
	file, err := os.Create(src)
	if err != nil {
		t.Fatal(err)
	}
	err = template.Must(template.New("main").Parse(testprog)).Execute(file, linktype != linkNoCgo)
	if err != nil {
		if err := file.Close(); err != nil {
			t.Error(err)
		}
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}

	exe := filepath.Join(tmpdir, "a.exe")
	args := []string{"build", "-o", exe}
	switch linktype {
	case linkNoCgo:
	case linkCgoDefault:
	case linkCgoInternal:
		args = append(args, "-ldflags", "-linkmode=internal")
	case linkCgoExternal:
		args = append(args, "-ldflags", "-linkmode=external")
	default:
		t.Fatalf("invalid linktype parameter of %v", linktype)
	}
	args = append(args, src)
	out, err := exec.Command(testenv.GoToolPath(t), args...).CombinedOutput()
	if err != nil {
		t.Fatalf("building test executable for linktype %d failed: %s %s", linktype, err, out)
	}
	out, err = exec.Command(exe).CombinedOutput()
	if err != nil {
		t.Fatalf("running test executable failed: %s %s", err, out)
	}

	matches := regexp.MustCompile("main=(.*)\n").FindStringSubmatch(string(out))
	if len(matches) < 2 {
		t.Fatalf("unexpected program output: %s", out)
	}
	wantaddr, err := strconv.ParseUint(matches[1], 0, 64)
	if err != nil {
		t.Fatalf("unexpected main address %q: %s", matches[1], err)
	}

	f, err := Open(exe)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	var foundDebugGDBScriptsSection bool
	for _, sect := range f.Sections {
		if sect.Name == ".debug_gdb_scripts" {
			foundDebugGDBScriptsSection = true
		}
	}
	if !foundDebugGDBScriptsSection {
		t.Error(".debug_gdb_scripts section is not found")
	}

	d, err := f.DWARF()
	if err != nil {
		t.Fatal(err)
	}

	// look for main.main
	r := d.Reader()
	for {
		e, err := r.Next()
		if err != nil {
			t.Fatal("r.Next:", err)
		}
		if e == nil {
			break
		}
		if e.Tag == dwarf.TagSubprogram {
			if name, ok := e.Val(dwarf.AttrName).(string); ok && name == "main.main" {
				if addr, ok := e.Val(dwarf.AttrLowpc).(uint64); ok && addr == wantaddr {
					return
				}
			}
		}
	}
	t.Fatal("main.main not found")
}

func TestBSSHasZeros(t *testing.T) {
	testenv.MustHaveExec(t)

	if runtime.GOOS != "windows" {
		t.Skip("skipping windows only test")
	}
	gccpath, err := exec.LookPath("gcc")
	if err != nil {
		t.Skip("skipping test: gcc is missing")
	}

	tmpdir, err := ioutil.TempDir("", "TestBSSHasZeros")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	srcpath := filepath.Join(tmpdir, "a.c")
	src := `
#include <stdio.h>

int zero = 0;

int
main(void)
{
	printf("%d\n", zero);
	return 0;
}
`
	err = ioutil.WriteFile(srcpath, []byte(src), 0644)
	if err != nil {
		t.Fatal(err)
	}

	objpath := filepath.Join(tmpdir, "a.obj")
	cmd := exec.Command(gccpath, "-c", srcpath, "-o", objpath)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build object file: %v - %v", err, string(out))
	}

	f, err := Open(objpath)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	var bss *Section
	for _, sect := range f.Sections {
		if sect.Name == ".bss" {
			bss = sect
			break
		}
	}
	if bss == nil {
		t.Fatal("could not find .bss section")
	}
	data, err := bss.Data()
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		t.Fatalf("%s file .bss section cannot be empty", objpath)
	}
	for _, b := range data {
		if b != 0 {
			t.Fatalf(".bss section has non zero bytes: %v", data)
		}
	}
}

func TestDWARF(t *testing.T) {
	testDWARF(t, linkNoCgo)
}

const testprog = `
package main

import "fmt"
{{if .}}import "C"
{{end}}

func main() {
	fmt.Printf("main=%p\n", main)
}
`

func TestBuildingWindowsGUI(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS != "windows" {
		t.Skip("skipping windows only test")
	}
	tmpdir, err := ioutil.TempDir("", "TestBuildingWindowsGUI")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	src := filepath.Join(tmpdir, "a.go")
	err = ioutil.WriteFile(src, []byte(`package main; func main() {}`), 0644)
	if err != nil {
		t.Fatal(err)
	}
	exe := filepath.Join(tmpdir, "a.exe")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-ldflags", "-H=windowsgui", "-o", exe, src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("building test executable failed: %s %s", err, out)
	}

	f, err := Open(exe)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	const _IMAGE_SUBSYSTEM_WINDOWS_GUI = 2

	switch oh := f.OptionalHeader.(type) {
	case *OptionalHeader32:
		if oh.Subsystem != _IMAGE_SUBSYSTEM_WINDOWS_GUI {
			t.Errorf("unexpected Subsystem value: have %d, but want %d", oh.Subsystem, _IMAGE_SUBSYSTEM_WINDOWS_GUI)
		}
	case *OptionalHeader64:
		if oh.Subsystem != _IMAGE_SUBSYSTEM_WINDOWS_GUI {
			t.Errorf("unexpected Subsystem value: have %d, but want %d", oh.Subsystem, _IMAGE_SUBSYSTEM_WINDOWS_GUI)
		}
	default:
		t.Fatalf("unexpected OptionalHeader type: have %T, but want *pe.OptionalHeader32 or *pe.OptionalHeader64", oh)
	}
}

func TestImportTableInUnknownSection(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("skipping windows only test")
	}

	// first we need to find this font driver
	path, err := exec.LookPath("atmfd.dll")
	if err != nil {
		t.Fatalf("unable to locate required file %q in search path: %s", "atmfd.dll", err)
	}

	f, err := Open(path)
	if err != nil {
		t.Error(err)
	}
	defer f.Close()

	// now we can extract its imports
	symbols, err := f.ImportedSymbols()
	if err != nil {
		t.Error(err)
	}

	if len(symbols) == 0 {
		t.Fatalf("unable to locate any imported symbols within file %q.", path)
	}
}
