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
		t.Skip("Disasm only tested on x86-64 linux")
	}
}

func TestDisasm(t *testing.T) {
	skipUnlessLinuxAmd64(t)
	bu := &Binutils{}
	insts, err := bu.Disasm(filepath.Join("testdata", "hello"), 0, math.MaxUint64)
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

func TestObjFile(t *testing.T) {
	skipUnlessLinuxAmd64(t)
	bu := &Binutils{}
	f, err := bu.Open(filepath.Join("testdata", "hello"), 0, math.MaxUint64, 0)
	if err != nil {
		t.Fatalf("Open: unexpected error %v", err)
	}
	defer f.Close()
	syms, err := f.Symbols(regexp.MustCompile("main"), 0)
	if err != nil {
		t.Fatalf("Symbols: unexpected error %v", err)
	}

	find := func(name string) *plugin.Sym {
		for _, s := range syms {
			for _, n := range s.Name {
				if n == name {
					return s
				}
			}
		}
		return nil
	}
	m := find("main")
	if m == nil {
		t.Fatalf("Symbols: did not find main")
	}
	frames, err := f.SourceLine(m.Start)
	if err != nil {
		t.Fatalf("SourceLine: unexpected error %v", err)
	}
	expect := []plugin.Frame{
		{Func: "main", File: "/tmp/hello.c", Line: 3},
	}
	if !reflect.DeepEqual(frames, expect) {
		t.Fatalf("SourceLine for main: expect %v; got %v\n", expect, frames)
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
