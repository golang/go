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
	"fmt"
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

	a := addr2Liner{&mockAddr2liner{}, offset, nil}
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
	oddSizedMap := addr2LinerNM{
		m: []symbolInfo{
			{0x1000, "0x1000"},
			{0x2000, "0x2000"},
			{0x3000, "0x3000"},
		},
	}
	evenSizedMap := addr2LinerNM{
		m: []symbolInfo{
			{0x1000, "0x1000"},
			{0x2000, "0x2000"},
			{0x3000, "0x3000"},
			{0x4000, "0x4000"},
		},
	}
	for _, a := range []*addr2LinerNM{
		&oddSizedMap, &evenSizedMap,
	} {
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
	}
}

func checkAddress(got []plugin.Frame, address uint64, want string) bool {
	if len(got) != 1 {
		return false
	}
	return got[0].Func == want
}
