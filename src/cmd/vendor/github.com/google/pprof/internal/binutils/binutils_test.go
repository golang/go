// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package binutils

import (
	"bytes"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strings"
	"testing"

	"github.com/google/pprof/internal/plugin"
)

var testAddrMap = map[int]string{
	1000: "_Z3fooid.clone2",
	2000: "_ZNSaIiEC1Ev.clone18",
	3000: "_ZNSt6vectorIS_IS_IiSaIiEESaIS1_EESaIS3_EEixEm",
}

func functionName(level int) (name string) {
	if name = testAddrMap[level]; name != "" {
		return name
	}
	return fmt.Sprintf("fun%d", level)
}

func TestAddr2Liner(t *testing.T) {
	const offset = 0x500

	a := addr2Liner{rw: &mockAddr2liner{}, base: offset}
	for i := 1; i < 8; i++ {
		addr := i*0x1000 + offset
		s, err := a.addrInfo(uint64(addr))
		if err != nil {
			t.Fatalf("addrInfo(%#x): %v", addr, err)
		}
		if len(s) != i {
			t.Fatalf("addrInfo(%#x): got len==%d, want %d", addr, len(s), i)
		}
		for l, f := range s {
			level := (len(s) - l) * 1000
			want := plugin.Frame{Func: functionName(level), File: fmt.Sprintf("file%d", level), Line: level}

			if f != want {
				t.Errorf("AddrInfo(%#x)[%d]: = %+v, want %+v", addr, l, f, want)
			}
		}
	}
	s, err := a.addrInfo(0xFFFF)
	if err != nil {
		t.Fatalf("addrInfo(0xFFFF): %v", err)
	}
	if len(s) != 0 {
		t.Fatalf("AddrInfo(0xFFFF): got len==%d, want 0", len(s))
	}
	a.rw.close()
}

type mockAddr2liner struct {
	output []string
}

func (a *mockAddr2liner) write(s string) error {
	var lines []string
	switch s {
	case "1000":
		lines = []string{"_Z3fooid.clone2", "file1000:1000"}
	case "2000":
		lines = []string{"_ZNSaIiEC1Ev.clone18", "file2000:2000", "_Z3fooid.clone2", "file1000:1000"}
	case "3000":
		lines = []string{"_ZNSt6vectorIS_IS_IiSaIiEESaIS1_EESaIS3_EEixEm", "file3000:3000", "_ZNSaIiEC1Ev.clone18", "file2000:2000", "_Z3fooid.clone2", "file1000:1000"}
	case "4000":
		lines = []string{"fun4000", "file4000:4000", "_ZNSt6vectorIS_IS_IiSaIiEESaIS1_EESaIS3_EEixEm", "file3000:3000", "_ZNSaIiEC1Ev.clone18", "file2000:2000", "_Z3fooid.clone2", "file1000:1000"}
	case "5000":
		lines = []string{"fun5000", "file5000:5000", "fun4000", "file4000:4000", "_ZNSt6vectorIS_IS_IiSaIiEESaIS1_EESaIS3_EEixEm", "file3000:3000", "_ZNSaIiEC1Ev.clone18", "file2000:2000", "_Z3fooid.clone2", "file1000:1000"}
	case "6000":
		lines = []string{"fun6000", "file6000:6000", "fun5000", "file5000:5000", "fun4000", "file4000:4000", "_ZNSt6vectorIS_IS_IiSaIiEESaIS1_EESaIS3_EEixEm", "file3000:3000", "_ZNSaIiEC1Ev.clone18", "file2000:2000", "_Z3fooid.clone2", "file1000:1000"}
	case "7000":
		lines = []string{"fun7000", "file7000:7000", "fun6000", "file6000:6000", "fun5000", "file5000:5000", "fun4000", "file4000:4000", "_ZNSt6vectorIS_IS_IiSaIiEESaIS1_EESaIS3_EEixEm", "file3000:3000", "_ZNSaIiEC1Ev.clone18", "file2000:2000", "_Z3fooid.clone2", "file1000:1000"}
	case "8000":
		lines = []string{"fun8000", "file8000:8000", "fun7000", "file7000:7000", "fun6000", "file6000:6000", "fun5000", "file5000:5000", "fun4000", "file4000:4000", "_ZNSt6vectorIS_IS_IiSaIiEESaIS1_EESaIS3_EEixEm", "file3000:3000", "_ZNSaIiEC1Ev.clone18", "file2000:2000", "_Z3fooid.clone2", "file1000:1000"}
	case "9000":
		lines = []string{"fun9000", "file9000:9000", "fun8000", "file8000:8000", "fun7000", "file7000:7000", "fun6000", "file6000:6000", "fun5000", "file5000:5000", "fun4000", "file4000:4000", "_ZNSt6vectorIS_IS_IiSaIiEESaIS1_EESaIS3_EEixEm", "file3000:3000", "_ZNSaIiEC1Ev.clone18", "file2000:2000", "_Z3fooid.clone2", "file1000:1000"}
	default:
		lines = []string{"??", "??:0"}
	}
	a.output = append(a.output, "0x"+s)
	a.output = append(a.output, lines...)
	return nil
}

func (a *mockAddr2liner) readLine() (string, error) {
	if len(a.output) == 0 {
		return "", fmt.Errorf("end of file")
	}
	next := a.output[0]
	a.output = a.output[1:]
	return next, nil
}

func (a *mockAddr2liner) close() {
}

func TestAddr2LinerLookup(t *testing.T) {
	const oddSizedData = `
00001000 T 0x1000
00002000 T 0x2000
00003000 T 0x3000
`
	const evenSizedData = `
0000000000001000 T 0x1000
0000000000002000 T 0x2000
0000000000003000 T 0x3000
0000000000004000 T 0x4000
`
	for _, d := range []string{oddSizedData, evenSizedData} {
		a, err := parseAddr2LinerNM(0, bytes.NewBufferString(d))
		if err != nil {
			t.Errorf("nm parse error: %v", err)
			continue
		}
		for address, want := range map[uint64]string{
			0x1000: "0x1000",
			0x1001: "0x1000",
			0x1FFF: "0x1000",
			0x2000: "0x2000",
			0x2001: "0x2000",
		} {
			if got, _ := a.addrInfo(address); !checkAddress(got, address, want) {
				t.Errorf("%x: got %v, want %s", address, got, want)
			}
		}
		for _, unknown := range []uint64{0x0fff, 0x4001} {
			if got, _ := a.addrInfo(unknown); got != nil {
				t.Errorf("%x: got %v, want nil", unknown, got)
			}
		}
	}
}

func checkAddress(got []plugin.Frame, address uint64, want string) bool {
	if len(got) != 1 {
		return false
	}
	return got[0].Func == want
}

func TestSetTools(t *testing.T) {
	// Test that multiple calls work.
	bu := &Binutils{}
	bu.SetTools("")
	bu.SetTools("")
}

func TestSetFastSymbolization(t *testing.T) {
	// Test that multiple calls work.
	bu := &Binutils{}
	bu.SetFastSymbolization(true)
	bu.SetFastSymbolization(false)
}

func skipUnlessLinuxAmd64(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skip("This test only works on x86-64 Linux")
	}
}

func skipUnlessDarwinAmd64(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "amd64" {
		t.Skip("This test only works on x86-64 Mac")
	}
}

func TestDisasm(t *testing.T) {
	skipUnlessLinuxAmd64(t)
	bu := &Binutils{}
	insts, err := bu.Disasm(filepath.Join("testdata", "exe_linux_64"), 0, math.MaxUint64)
	if err != nil {
		t.Fatalf("Disasm: unexpected error %v", err)
	}
	mainCount := 0
	for _, x := range insts {
		if x.Function == "main" {
			mainCount++
		}
	}
	if mainCount == 0 {
		t.Error("Disasm: found no main instructions")
	}
}

func findSymbol(syms []*plugin.Sym, name string) *plugin.Sym {
	for _, s := range syms {
		for _, n := range s.Name {
			if n == name {
				return s
			}
		}
	}
	return nil
}

func TestObjFile(t *testing.T) {
	skipUnlessLinuxAmd64(t)
	for _, tc := range []struct {
		desc                 string
		start, limit, offset uint64
		addr                 uint64
	}{
		{"fake mapping", 0, math.MaxUint64, 0, 0x40052d},
		{"fixed load address", 0x400000, 0x4006fc, 0, 0x40052d},
		// True user-mode ASLR binaries are ET_DYN rather than ET_EXEC so this case
		// is a bit artificial except that it approximates the
		// vmlinux-with-kernel-ASLR case where the binary *is* ET_EXEC.
		{"simulated ASLR address", 0x500000, 0x5006fc, 0, 0x50052d},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			bu := &Binutils{}
			f, err := bu.Open(filepath.Join("testdata", "exe_linux_64"), tc.start, tc.limit, tc.offset)
			if err != nil {
				t.Fatalf("Open: unexpected error %v", err)
			}
			defer f.Close()
			syms, err := f.Symbols(regexp.MustCompile("main"), 0)
			if err != nil {
				t.Fatalf("Symbols: unexpected error %v", err)
			}

			m := findSymbol(syms, "main")
			if m == nil {
				t.Fatalf("Symbols: did not find main")
			}
			for _, addr := range []uint64{m.Start + f.Base(), tc.addr} {
				gotFrames, err := f.SourceLine(addr)
				if err != nil {
					t.Fatalf("SourceLine: unexpected error %v", err)
				}
				wantFrames := []plugin.Frame{
					{Func: "main", File: "/tmp/hello.c", Line: 3},
				}
				if !reflect.DeepEqual(gotFrames, wantFrames) {
					t.Fatalf("SourceLine for main: got %v; want %v\n", gotFrames, wantFrames)
				}
			}
		})
	}
}

func TestMachoFiles(t *testing.T) {
	skipUnlessDarwinAmd64(t)

	// Load `file`, pretending it was mapped at `start`. Then get the symbol
	// table. Check that it contains the symbol `sym` and that the address
	// `addr` gives the `expected` stack trace.
	for _, tc := range []struct {
		desc                 string
		file                 string
		start, limit, offset uint64
		addr                 uint64
		sym                  string
		expected             []plugin.Frame
	}{
		{"normal mapping", "exe_mac_64", 0x100000000, math.MaxUint64, 0,
			0x100000f50, "_main",
			[]plugin.Frame{
				{Func: "main", File: "/tmp/hello.c", Line: 3},
			}},
		{"other mapping", "exe_mac_64", 0x200000000, math.MaxUint64, 0,
			0x200000f50, "_main",
			[]plugin.Frame{
				{Func: "main", File: "/tmp/hello.c", Line: 3},
			}},
		{"lib normal mapping", "lib_mac_64", 0, math.MaxUint64, 0,
			0xfa0, "_bar",
			[]plugin.Frame{
				{Func: "bar", File: "/tmp/lib.c", Line: 5},
			}},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			bu := &Binutils{}
			f, err := bu.Open(filepath.Join("testdata", tc.file), tc.start, tc.limit, tc.offset)
			if err != nil {
				t.Fatalf("Open: unexpected error %v", err)
			}
			t.Logf("binutils: %v", bu)
			if runtime.GOOS == "darwin" && !bu.rep.addr2lineFound && !bu.rep.llvmSymbolizerFound {
				// On OSX user needs to install gaddr2line or llvm-symbolizer with
				// Homebrew, skip the test when the environment doesn't have it
				// installed.
				t.Skip("couldn't find addr2line or gaddr2line")
			}
			defer f.Close()
			syms, err := f.Symbols(nil, 0)
			if err != nil {
				t.Fatalf("Symbols: unexpected error %v", err)
			}

			m := findSymbol(syms, tc.sym)
			if m == nil {
				t.Fatalf("Symbols: could not find symbol %v", tc.sym)
			}
			gotFrames, err := f.SourceLine(tc.addr)
			if err != nil {
				t.Fatalf("SourceLine: unexpected error %v", err)
			}
			if !reflect.DeepEqual(gotFrames, tc.expected) {
				t.Fatalf("SourceLine for main: got %v; want %v\n", gotFrames, tc.expected)
			}
		})
	}
}

func TestLLVMSymbolizer(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("testtdata/llvm-symbolizer has only been tested on linux")
	}

	cmd := filepath.Join("testdata", "fake-llvm-symbolizer")
	symbolizer, err := newLLVMSymbolizer(cmd, "foo", 0)
	if err != nil {
		t.Fatalf("newLLVMSymbolizer: unexpected error %v", err)
	}
	defer symbolizer.rw.close()

	for _, c := range []struct {
		addr   uint64
		frames []plugin.Frame
	}{
		{0x10, []plugin.Frame{
			{Func: "Inlined_0x10", File: "foo.h", Line: 0},
			{Func: "Func_0x10", File: "foo.c", Line: 2},
		}},
		{0x20, []plugin.Frame{
			{Func: "Inlined_0x20", File: "foo.h", Line: 0},
			{Func: "Func_0x20", File: "foo.c", Line: 2},
		}},
	} {
		frames, err := symbolizer.addrInfo(c.addr)
		if err != nil {
			t.Errorf("LLVM: unexpected error %v", err)
			continue
		}
		if !reflect.DeepEqual(frames, c.frames) {
			t.Errorf("LLVM: expect %v; got %v\n", c.frames, frames)
		}
	}
}

func TestOpenMalformedELF(t *testing.T) {
	// Test that opening a malformed ELF file will report an error containing
	// the word "ELF".
	bu := &Binutils{}
	_, err := bu.Open(filepath.Join("testdata", "malformed_elf"), 0, 0, 0)
	if err == nil {
		t.Fatalf("Open: unexpected success")
	}

	if !strings.Contains(err.Error(), "ELF") {
		t.Errorf("Open: got %v, want error containing 'ELF'", err)
	}
}

func TestOpenMalformedMachO(t *testing.T) {
	// Test that opening a malformed Mach-O file will report an error containing
	// the word "Mach-O".
	bu := &Binutils{}
	_, err := bu.Open(filepath.Join("testdata", "malformed_macho"), 0, 0, 0)
	if err == nil {
		t.Fatalf("Open: unexpected success")
	}

	if !strings.Contains(err.Error(), "Mach-O") {
		t.Errorf("Open: got %v, want error containing 'Mach-O'", err)
	}
}
