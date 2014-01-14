// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for auto-generated symbols.

// There is no test for $f64. and $f32. symbols, because those are
// not possible to write in the assembler syntax. Instead of changing
// the assembler to allow that, we plan to change the compilers
// not to generate such symbols (plain dupok data is sufficient).

package main

import (
	"bytes"
	"debug/goobj"
	"testing"
)

// Each test case is an object file, generated from a corresponding .s file.
// The image of the autotab symbol should be a sequence of pairs of
// identical 8-byte sequences.
var autoTests = []string{
	"testdata/autosection.6",
	"testdata/autoweak.6",
}

func TestAuto(t *testing.T) {
	for _, obj := range autoTests {
		p := Prog{GOOS: "darwin", GOARCH: "amd64", StartSym: "start"}
		p.omitRuntime = true
		p.Error = func(s string) { t.Error(s) }
		var buf bytes.Buffer
		p.link(&buf, obj)
		if p.NumError > 0 {
			continue // already reported
		}

		const name = "autotab"
		sym := p.Syms[goobj.SymID{Name: name}]
		if sym == nil {
			t.Errorf("%s is missing %s symbol", obj, name)
			return
		}
		if sym.Size == 0 {
			return
		}

		seg := sym.Section.Segment
		off := sym.Addr - seg.VirtAddr
		data := seg.Data[off : off+Addr(sym.Size)]
		if len(data)%16 != 0 {
			t.Errorf("%s: %s.Size = %d, want multiple of 16", obj, name, len(data))
			return
		}
	Data:
		for i := 0; i < len(data); i += 16 {
			have := p.byteorder.Uint64(data[i : i+8])
			want := p.byteorder.Uint64(data[i+8 : i+16])
			if have != want {
				// Look for relocation so we can explain what went wrong.
				for _, r := range sym.Reloc {
					if r.Offset == i {
						t.Errorf("%s: %s+%#x: %s: have %#x want %#x", obj, name, i, r.Sym, have, want)
						continue Data
					}
				}
				t.Errorf("%s: %s+%#x: have %#x want %#x", obj, name, i, have, want)
			}
		}
	}
}
