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
	"regexp"
	"testing"

	"github.com/google/pprof/internal/plugin"
)

// TestFindSymbols tests the FindSymbols routine using a hardcoded nm output.
func TestFindSymbols(t *testing.T) {
	type testcase struct {
		query, syms string
		want        []plugin.Sym
	}

	testsyms := `0000000000001000 t lineA001
0000000000001000 t lineA002
0000000000001000 t line1000
0000000000002000 t line200A
0000000000002000 t line2000
0000000000002000 t line200B
0000000000003000 t line3000
0000000000003000 t _ZNK4DumbclEPKc
0000000000003000 t lineB00C
0000000000003000 t line300D
0000000000004000 t _the_end
	`
	testcases := []testcase{
		{
			"line.*[AC]",
			testsyms,
			[]plugin.Sym{
				{Name: []string{"lineA001"}, File: "object.o", Start: 0x1000, End: 0x1FFF},
				{Name: []string{"line200A"}, File: "object.o", Start: 0x2000, End: 0x2FFF},
				{Name: []string{"lineB00C"}, File: "object.o", Start: 0x3000, End: 0x3FFF},
			},
		},
		{
			"Dumb::operator",
			testsyms,
			[]plugin.Sym{
				{Name: []string{"Dumb::operator()(char const*) const"}, File: "object.o", Start: 0x3000, End: 0x3FFF},
			},
		},
	}

	for _, tc := range testcases {
		syms, err := findSymbols([]byte(tc.syms), "object.o", regexp.MustCompile(tc.query), 0)
		if err != nil {
			t.Fatalf("%q: findSymbols: %v", tc.query, err)
		}
		if err := checkSymbol(syms, tc.want); err != nil {
			t.Errorf("%q: %v", tc.query, err)
		}
	}
}

func checkSymbol(got []*plugin.Sym, want []plugin.Sym) error {
	if len(got) != len(want) {
		return fmt.Errorf("unexpected number of symbols %d (want %d)\n", len(got), len(want))
	}

	for i, g := range got {
		w := want[i]
		if len(g.Name) != len(w.Name) {
			return fmt.Errorf("names, got %d, want %d", len(g.Name), len(w.Name))
		}
		for n := range g.Name {
			if g.Name[n] != w.Name[n] {
				return fmt.Errorf("name %d, got %q, want %q", n, g.Name[n], w.Name[n])
			}
		}
		if g.File != w.File {
			return fmt.Errorf("filename, got %q, want %q", g.File, w.File)
		}
		if g.Start != w.Start {
			return fmt.Errorf("start address, got %#x, want %#x", g.Start, w.Start)
		}
		if g.End != w.End {
			return fmt.Errorf("end address, got %#x, want %#x", g.End, w.End)
		}
	}
	return nil
}

// TestFunctionAssembly tests the FunctionAssembly routine by using a
// fake objdump script.
func TestFunctionAssembly(t *testing.T) {
	type testcase struct {
		s    plugin.Sym
		asm  string
		want []plugin.Inst
	}
	testcases := []testcase{
		{
			plugin.Sym{Name: []string{"symbol1"}, Start: 0x1000, End: 0x1FFF},
			`  1000: instruction one
  1001: instruction two
  1002: instruction three
  1003: instruction four
`,
			[]plugin.Inst{
				{Addr: 0x1000, Text: "instruction one"},
				{Addr: 0x1001, Text: "instruction two"},
				{Addr: 0x1002, Text: "instruction three"},
				{Addr: 0x1003, Text: "instruction four"},
			},
		},
		{
			plugin.Sym{Name: []string{"symbol2"}, Start: 0x2000, End: 0x2FFF},
			`  2000: instruction one
  2001: instruction two
`,
			[]plugin.Inst{
				{Addr: 0x2000, Text: "instruction one"},
				{Addr: 0x2001, Text: "instruction two"},
			},
		},
	}

	const objdump = "testdata/wrapper/objdump"

	for _, tc := range testcases {
		insts, err := disassemble([]byte(tc.asm))
		if err != nil {
			t.Fatalf("FunctionAssembly: %v", err)
		}

		if len(insts) != len(tc.want) {
			t.Errorf("Unexpected number of assembly instructions %d (want %d)\n", len(insts), len(tc.want))
		}
		for i := range insts {
			if insts[i] != tc.want[i] {
				t.Errorf("Expected symbol %v, got %v\n", tc.want[i], insts[i])
			}
		}
	}
}
