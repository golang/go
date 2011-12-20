// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gosym

import (
	"debug/elf"
	"os"
	"syscall"
	"testing"
)

func dotest() bool {
	// For now, only works on ELF platforms.
	// TODO: convert to work with new go tool
	return false && syscall.OS == "linux" && os.Getenv("GOARCH") == "amd64"
}

func getTable(t *testing.T) *Table {
	f, tab := crack(os.Args[0], t)
	f.Close()
	return tab
}

func crack(file string, t *testing.T) (*elf.File, *Table) {
	// Open self
	f, err := elf.Open(file)
	if err != nil {
		t.Fatal(err)
	}
	return parse(file, f, t)
}

func parse(file string, f *elf.File, t *testing.T) (*elf.File, *Table) {
	symdat, err := f.Section(".gosymtab").Data()
	if err != nil {
		f.Close()
		t.Fatalf("reading %s gosymtab: %v", file, err)
	}
	pclndat, err := f.Section(".gopclntab").Data()
	if err != nil {
		f.Close()
		t.Fatalf("reading %s gopclntab: %v", file, err)
	}

	pcln := NewLineTable(pclndat, f.Section(".text").Addr)
	tab, err := NewTable(symdat, pcln)
	if err != nil {
		f.Close()
		t.Fatalf("parsing %s gosymtab: %v", file, err)
	}

	return f, tab
}

var goarch = os.Getenv("O")

func TestLineFromAline(t *testing.T) {
	if !dotest() {
		return
	}

	tab := getTable(t)

	// Find the sym package
	pkg := tab.LookupFunc("debug/gosym.TestLineFromAline").Obj
	if pkg == nil {
		t.Fatalf("nil pkg")
	}

	// Walk every absolute line and ensure that we hit every
	// source line monotonically
	lastline := make(map[string]int)
	final := -1
	for i := 0; i < 10000; i++ {
		path, line := pkg.lineFromAline(i)
		// Check for end of object
		if path == "" {
			if final == -1 {
				final = i - 1
			}
			continue
		} else if final != -1 {
			t.Fatalf("reached end of package at absolute line %d, but absolute line %d mapped to %s:%d", final, i, path, line)
		}
		// It's okay to see files multiple times (e.g., sys.a)
		if line == 1 {
			lastline[path] = 1
			continue
		}
		// Check that the is the next line in path
		ll, ok := lastline[path]
		if !ok {
			t.Errorf("file %s starts on line %d", path, line)
		} else if line != ll+1 {
			t.Errorf("expected next line of file %s to be %d, got %d", path, ll+1, line)
		}
		lastline[path] = line
	}
	if final == -1 {
		t.Errorf("never reached end of object")
	}
}

func TestLineAline(t *testing.T) {
	if !dotest() {
		return
	}

	tab := getTable(t)

	for _, o := range tab.Files {
		// A source file can appear multiple times in a
		// object.  alineFromLine will always return alines in
		// the first file, so track which lines we've seen.
		found := make(map[string]int)
		for i := 0; i < 1000; i++ {
			path, line := o.lineFromAline(i)
			if path == "" {
				break
			}

			// cgo files are full of 'Z' symbols, which we don't handle
			if len(path) > 4 && path[len(path)-4:] == ".cgo" {
				continue
			}

			if minline, ok := found[path]; path != "" && ok {
				if minline >= line {
					// We've already covered this file
					continue
				}
			}
			found[path] = line

			a, err := o.alineFromLine(path, line)
			if err != nil {
				t.Errorf("absolute line %d in object %s maps to %s:%d, but mapping that back gives error %s", i, o.Paths[0].Name, path, line, err)
			} else if a != i {
				t.Errorf("absolute line %d in object %s maps to %s:%d, which maps back to absolute line %d\n", i, o.Paths[0].Name, path, line, a)
			}
		}
	}
}

func TestPCLine(t *testing.T) {
	if !dotest() {
		return
	}

	f, tab := crack("_test/pclinetest", t)
	text := f.Section(".text")
	textdat, err := text.Data()
	if err != nil {
		t.Fatalf("reading .text: %v", err)
	}

	// Test PCToLine
	sym := tab.LookupFunc("linefrompc")
	wantLine := 0
	for pc := sym.Entry; pc < sym.End; pc++ {
		file, line, fn := tab.PCToLine(pc)
		off := pc - text.Addr // TODO(rsc): should not need off; bug in 8g
		wantLine += int(textdat[off])
		if fn == nil {
			t.Errorf("failed to get line of PC %#x", pc)
		} else if len(file) < 12 || file[len(file)-12:] != "pclinetest.s" || line != wantLine || fn != sym {
			t.Errorf("expected %s:%d (%s) at PC %#x, got %s:%d (%s)", "pclinetest.s", wantLine, sym.Name, pc, file, line, fn.Name)
		}
	}

	// Test LineToPC
	sym = tab.LookupFunc("pcfromline")
	lookupline := -1
	wantLine = 0
	off := uint64(0) // TODO(rsc): should not need off; bug in 8g
	for pc := sym.Value; pc < sym.End; pc += 2 + uint64(textdat[off]) {
		file, line, fn := tab.PCToLine(pc)
		off = pc - text.Addr
		wantLine += int(textdat[off])
		if line != wantLine {
			t.Errorf("expected line %d at PC %#x in pcfromline, got %d", wantLine, pc, line)
			off = pc + 1 - text.Addr
			continue
		}
		if lookupline == -1 {
			lookupline = line
		}
		for ; lookupline <= line; lookupline++ {
			pc2, fn2, err := tab.LineToPC(file, lookupline)
			if lookupline != line {
				// Should be nothing on this line
				if err == nil {
					t.Errorf("expected no PC at line %d, got %#x (%s)", lookupline, pc2, fn2.Name)
				}
			} else if err != nil {
				t.Errorf("failed to get PC of line %d: %s", lookupline, err)
			} else if pc != pc2 {
				t.Errorf("expected PC %#x (%s) at line %d, got PC %#x (%s)", pc, fn.Name, line, pc2, fn2.Name)
			}
		}
		off = pc + 1 - text.Addr
	}
}
