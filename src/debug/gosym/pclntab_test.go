// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gosym

import (
	"debug/elf"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

var (
	pclineTempDir    string
	pclinetestBinary string
)

func dotest(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	// For now, only works on amd64 platforms.
	if runtime.GOARCH != "amd64" {
		t.Skipf("skipping on non-AMD64 system %s", runtime.GOARCH)
	}
	var err error
	pclineTempDir, err = ioutil.TempDir("", "pclinetest")
	if err != nil {
		t.Fatal(err)
	}
	// This command builds pclinetest from pclinetest.asm;
	// the resulting binary looks like it was built from pclinetest.s,
	// but we have renamed it to keep it away from the go tool.
	pclinetestBinary = filepath.Join(pclineTempDir, "pclinetest")
	cmd := exec.Command("go", "tool", "asm", "-o", pclinetestBinary+".o", "pclinetest.asm")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatal(err)
	}
	cmd = exec.Command("go", "tool", "link", "-H", "linux", "-E", "main",
		"-o", pclinetestBinary, pclinetestBinary+".o")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		t.Fatal(err)
	}
}

func endtest() {
	if pclineTempDir != "" {
		os.RemoveAll(pclineTempDir)
		pclineTempDir = ""
		pclinetestBinary = ""
	}
}

// skipIfNotELF skips the test if we are not running on an ELF system.
// These tests open and examine the test binary, and use elf.Open to do so.
func skipIfNotELF(t *testing.T) {
	switch runtime.GOOS {
	case "dragonfly", "freebsd", "linux", "netbsd", "openbsd", "solaris":
		// OK.
	default:
		t.Skipf("skipping on non-ELF system %s", runtime.GOOS)
	}
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
	s := f.Section(".gosymtab")
	if s == nil {
		t.Skip("no .gosymtab section")
	}
	symdat, err := s.Data()
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
	skipIfNotELF(t)

	tab := getTable(t)
	if tab.go12line != nil {
		// aline's don't exist in the Go 1.2 table.
		t.Skip("not relevant to Go 1.2 symbol table")
	}

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
			t.Fatalf("expected next line of file %s to be %d, got %d", path, ll+1, line)
		}
		lastline[path] = line
	}
	if final == -1 {
		t.Errorf("never reached end of object")
	}
}

func TestLineAline(t *testing.T) {
	skipIfNotELF(t)

	tab := getTable(t)
	if tab.go12line != nil {
		// aline's don't exist in the Go 1.2 table.
		t.Skip("not relevant to Go 1.2 symbol table")
	}

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
	dotest(t)
	defer endtest()

	f, tab := crack(pclinetestBinary, t)
	text := f.Section(".text")
	textdat, err := text.Data()
	if err != nil {
		t.Fatalf("reading .text: %v", err)
	}

	// Test PCToLine
	sym := tab.LookupFunc("linefrompc")
	wantLine := 0
	for pc := sym.Entry; pc < sym.End; pc++ {
		off := pc - text.Addr // TODO(rsc): should not need off; bug in 8g
		if textdat[off] == 255 {
			break
		}
		wantLine += int(textdat[off])
		t.Logf("off is %d %#x (max %d)", off, textdat[off], sym.End-pc)
		file, line, fn := tab.PCToLine(pc)
		if fn == nil {
			t.Errorf("failed to get line of PC %#x", pc)
		} else if !strings.HasSuffix(file, "pclinetest.asm") || line != wantLine || fn != sym {
			t.Errorf("PCToLine(%#x) = %s:%d (%s), want %s:%d (%s)", pc, file, line, fn.Name, "pclinetest.asm", wantLine, sym.Name)
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
		if textdat[off] == 255 {
			break
		}
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
