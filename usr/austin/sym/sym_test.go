// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sym

import (
	"exec";
	"io";
	"os";
	"testing";
	"syscall";
)

var goarch = os.Getenv("O")
// No ELF binaries on OS X
var darwin = syscall.OS == "darwin";

func TestLineFromAline(t *testing.T) {
	if darwin {
		return;
	}

	// Use myself for this test
	f, err := os.Open(goarch + ".out", os.O_RDONLY, 0);
	if err != nil {
		t.Fatalf("failed to open %s.out: %s", goarch, err);
	}

	elf, err := NewElf(f);
	if err != nil {
		t.Fatalf("failed to read ELF: %s", err);
	}

	syms, err := ElfGoSyms(elf);
	if err != nil {
		t.Fatalf("failed to load syms: %s", err);
	}

	// Find the sym package
	pkg := syms.SymFromName("sym·ElfGoSyms").(*TextSym).obj;

	// Walk every absolute line and ensure that we hit every
	// source line monotonically
	lastline := make(map[string] int);
	final := -1;
	for i := 0; i < 10000; i++ {
		path, line := pkg.lineFromAline(i);
		// Check for end of object
		if path == "" {
			if final == -1 {
				final = i - 1;
			}
			continue;
		} else if final != -1 {
			t.Fatalf("reached end of package at absolute line %d, but absolute line %d mapped to %s:%d", final, i, path, line);
		}
		// It's okay to see files multiple times (e.g., sys.a)
		if line == 1 {
			lastline[path] = 1;
			continue;
		}
		// Check that the is the next line in path
		ll, ok := lastline[path];
		if !ok {
			t.Errorf("file %s starts on line %d", path, line);
		} else if line != ll + 1 {
			t.Errorf("expected next line of file %s to be %d, got %d", path, ll + 1, line);
		}
		lastline[path] = line;
	}
	if final == -1 {
		t.Errorf("never reached end of object");
	}
}

func TestLineAline(t *testing.T) {
	if darwin {
		return;
	}

	// Use myself for this test
	f, err := os.Open(goarch + ".out", os.O_RDONLY, 0);
	if err != nil {
		t.Fatalf("failed to open %s.out: %s", goarch, err);
	}

	elf, err := NewElf(f);
	if err != nil {
		t.Fatalf("failed to read ELF: %s", err);
	}

	syms, err := ElfGoSyms(elf);
	if err != nil {
		t.Fatalf("failed to load syms: %s", err);
	}

	for _, o := range syms.files {
		// A source file can appear multiple times in a
		// object.  alineFromLine will always return alines in
		// the first file, so track which lines we've seen.
		found := make(map[string] int);
		for i := 0; i < 1000; i++ {
			path, line := o.lineFromAline(i);
			if path == "" {
				break;
			}

			// cgo files are full of 'Z' symbols, which we don't handle
			if len(path) > 4 && path[len(path)-4:len(path)] == ".cgo" {
				continue;
			}

			if minline, ok := found[path]; path != "" && ok {
				if minline >= line {
					// We've already covered this file
					continue;
				}
			}
			found[path] = line;

			a, err := o.alineFromLine(path, line);
			if err != nil {
				t.Errorf("absolute line %d in object %s maps to %s:%d, but mapping that back gives error %s", i, o.paths[0].Name, path, line, err);
			} else if a != i {
				t.Errorf("absolute line %d in object %s maps to %s:%d, which maps back to absolute line %d\n", i, o.paths[0].Name, path, line, a);
			}
		}
	}
}

// gotest: if [ "`uname`" != "Darwin" ]; then
// gotest:    mkdir -p _test && $AS pclinetest.s && $LD -E main -l -o _test/pclinetest pclinetest.$O
// gotest: fi
func TestPCLine(t *testing.T) {
	if darwin {
		return;
	}

	f, err := os.Open("_test/pclinetest", os.O_RDONLY, 0);
	if err != nil {
		t.Fatalf("failed to open pclinetest.6: %s", err);
	}
	defer f.Close();

	elf, err := NewElf(f);
	if err != nil {
		t.Fatalf("failed to read ELF: %s", err);
	}

	syms, err := ElfGoSyms(elf);
	if err != nil {
		t.Fatalf("failed to load syms: %s", err);
	}

	textSec := elf.Section(".text");
	sf, err := textSec.Open();
	if err != nil {
		t.Fatalf("failed to open .text section: %s", err);
	}
	text, err := io.ReadAll(sf);
	if err != nil {
		t.Fatalf("failed to read .text section: %s", err);
	}

	// Test LineFromPC
	sym := syms.SymFromName("linefrompc").(*TextSym);
	wantLine := 0;
	for pc := sym.Value; pc < sym.End; pc++ {
		file, line, fn := syms.LineFromPC(pc);
		wantLine += int(text[pc-textSec.Addr]);
		if fn == nil {
			t.Errorf("failed to get line of PC %#x", pc);
		} else if len(file) < 12 || file[len(file)-12:len(file)] != "pclinetest.s" || line != wantLine || fn != sym {
			t.Errorf("expected %s:%d (%s) at PC %#x, got %s:%d (%s)", "pclinetest.s", wantLine, sym.Name, pc, file, line, fn.Name);
		}
	}

	// Test PCFromLine
	sym = syms.SymFromName("pcfromline").(*TextSym);
	lookupline := -1;
	wantLine = 0;
	for pc := sym.Value; pc < sym.End; pc += 2 + uint64(text[pc+1-textSec.Addr]) {
		file, line, fn := syms.LineFromPC(pc);
		wantLine += int(text[pc-textSec.Addr]);
		if line != wantLine {
			t.Errorf("expected line %d at PC %#x in pcfromline, got %d", wantLine, pc, line);
			continue;
		}
		if lookupline == -1 {
			lookupline = line;
		}
		for ; lookupline <= line; lookupline++ {
			pc2, fn2, err := syms.PCFromLine(file, lookupline);
			if lookupline != line {
				// Should be nothing on this line
				if err == nil {
					t.Errorf("expected no PC at line %d, got %#x (%s)", lookupline, pc2, fn2.Name);
				}
			} else if err != nil {
				t.Errorf("failed to get PC of line %d: %s", lookupline, err);
			} else if pc != pc2 {
				t.Errorf("expected PC %#x (%s) at line %d, got PC %#x (%s)", pc, fn.Name, line, pc2, fn2.Name);
			}
		}
	}
}
