// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	"fmt"
	"testing"
)

type testpoint struct {
	name   string
	ok     bool
	err    string
	auxstr string
}

func TestReadCOFFSymbolAuxInfo(t *testing.T) {
	testpoints := map[int]testpoint{
		39: {
			name:   ".rdata$.refptr.__native_startup_lock",
			ok:     true,
			auxstr: "{Size:8 NumRelocs:1 NumLineNumbers:0 Checksum:0 SecNum:16 Selection:2 _:[0 0 0]}",
		},
		81: {
			name:   ".debug_line",
			ok:     true,
			auxstr: "{Size:994 NumRelocs:1 NumLineNumbers:0 Checksum:1624223678 SecNum:32 Selection:0 _:[0 0 0]}",
		},
		155: {
			name: ".file",
			ok:   false,
			err:  "incorrect symbol storage class",
		},
	}

	// The testdata PE object file below was selected from a release
	// build from https://github.com/mstorsjo/llvm-mingw/releases; it
	// corresponds to the mingw "crt2.o" object. The object itself was
	// built using an x86_64 HOST=linux TARGET=windows clang cross
	// compiler based on LLVM 13. More build details can be found at
	// https://github.com/mstorsjo/llvm-mingw/releases.
	f, err := Open("testdata/llvm-mingw-20211002-msvcrt-x86_64-crt2")
	if err != nil {
		t.Errorf("open failed with %v", err)
	}
	defer f.Close()
	for k := range f.COFFSymbols {
		tp, ok := testpoints[k]
		if !ok {
			continue
		}
		sym := &f.COFFSymbols[k]
		if sym.NumberOfAuxSymbols == 0 {
			t.Errorf("expected aux symbols for sym %d", k)
			continue
		}
		name, nerr := sym.FullName(f.StringTable)
		if nerr != nil {
			t.Errorf("FullName(%d) failed with %v", k, nerr)
			continue
		}
		if name != tp.name {
			t.Errorf("name check for %d, got %s want %s", k, name, tp.name)
			continue
		}
		ap, err := f.COFFSymbolReadSectionDefAux(k)
		if tp.ok {
			if err != nil {
				t.Errorf("unexpected failure on %d, got error %v", k, err)
				continue
			}
			got := fmt.Sprintf("%+v", *ap)
			if got != tp.auxstr {
				t.Errorf("COFFSymbolReadSectionDefAux on %d bad return, got:\n%s\nwant:\n%s\n", k, got, tp.auxstr)
				continue
			}
		} else {
			if err == nil {
				t.Errorf("unexpected non-failure on %d", k)
				continue
			}
			got := fmt.Sprintf("%v", err)
			if got != tp.err {
				t.Errorf("COFFSymbolReadSectionDefAux %d wrong error, got %q want %q", k, got, tp.err)
				continue
			}
		}
	}
}
