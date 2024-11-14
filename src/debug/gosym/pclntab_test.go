// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gosym

import (
	"bytes"
	"compress/gzip"
	"debug/elf"
	"internal/testenv"
	"io"
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
	// This test builds a Linux/AMD64 binary. Skipping in short mode if cross compiling.
	if runtime.GOOS != "linux" && testing.Short() {
		t.Skipf("skipping in short mode on non-Linux system %s", runtime.GOARCH)
	}
	var err error
	pclineTempDir, err = os.MkdirTemp("", "pclinetest")
	if err != nil {
		t.Fatal(err)
	}
	pclinetestBinary = filepath.Join(pclineTempDir, "pclinetest")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", pclinetestBinary)
	cmd.Dir = "testdata"
	cmd.Env = append(os.Environ(), "GOOS=linux")
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
	case "dragonfly", "freebsd", "linux", "netbsd", "openbsd", "solaris", "illumos":
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
	defer f.Close()
	text := f.Section(".text")
	textdat, err := text.Data()
	if err != nil {
		t.Fatalf("reading .text: %v", err)
	}

	// Test PCToLine
	sym := tab.LookupFunc("main.linefrompc")
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
		} else if !strings.HasSuffix(file, "pclinetest.s") || line != wantLine || fn != sym {
			t.Errorf("PCToLine(%#x) = %s:%d (%s), want %s:%d (%s)", pc, file, line, fn.Name, "pclinetest.s", wantLine, sym.Name)
		}
	}

	// Test LineToPC
	sym = tab.LookupFunc("main.pcfromline")
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

func TestSymVersion(t *testing.T) {
	skipIfNotELF(t)

	table := getTable(t)
	if table.go12line == nil {
		t.Skip("not relevant to Go 1.2+ symbol table")
	}
	for _, fn := range table.Funcs {
		if fn.goVersion == verUnknown {
			t.Fatalf("unexpected symbol version: %v", fn)
		}
	}
}

// read115Executable returns a hello world executable compiled by Go 1.15.
//
// The file was compiled in /tmp/hello.go:
//
//	package main
//
//	func main() {
//		println("hello")
//	}
func read115Executable(tb testing.TB) []byte {
	zippedDat, err := os.ReadFile("testdata/pcln115.gz")
	if err != nil {
		tb.Fatal(err)
	}
	var gzReader *gzip.Reader
	gzReader, err = gzip.NewReader(bytes.NewBuffer(zippedDat))
	if err != nil {
		tb.Fatal(err)
	}
	var dat []byte
	dat, err = io.ReadAll(gzReader)
	if err != nil {
		tb.Fatal(err)
	}
	return dat
}

// Test that we can parse a pclntab from 1.15.
func Test115PclnParsing(t *testing.T) {
	dat := read115Executable(t)
	const textStart = 0x1001000
	pcln := NewLineTable(dat, textStart)
	tab, err := NewTable(nil, pcln)
	if err != nil {
		t.Fatal(err)
	}
	var f *Func
	var pc uint64
	pc, f, err = tab.LineToPC("/tmp/hello.go", 3)
	if err != nil {
		t.Fatal(err)
	}
	if pcln.version != ver12 {
		t.Fatal("Expected pcln to parse as an older version")
	}
	if pc != 0x105c280 {
		t.Fatalf("expect pc = 0x105c280, got 0x%x", pc)
	}
	if f.Name != "main.main" {
		t.Fatalf("expected to parse name as main.main, got %v", f.Name)
	}
}

var (
	sinkLineTable *LineTable
	sinkTable     *Table
)

func Benchmark115(b *testing.B) {
	dat := read115Executable(b)
	const textStart = 0x1001000

	b.Run("NewLineTable", func { b ->
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			sinkLineTable = NewLineTable(dat, textStart)
		}
	})

	pcln := NewLineTable(dat, textStart)
	b.Run("NewTable", func { b ->
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			var err error
			sinkTable, err = NewTable(nil, pcln)
			if err != nil {
				b.Fatal(err)
			}
		}
	})

	tab, err := NewTable(nil, pcln)
	if err != nil {
		b.Fatal(err)
	}

	b.Run("LineToPC", func { b ->
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			var f *Func
			var pc uint64
			pc, f, err = tab.LineToPC("/tmp/hello.go", 3)
			if err != nil {
				b.Fatal(err)
			}
			if pcln.version != ver12 {
				b.Fatalf("want version=%d, got %d", ver12, pcln.version)
			}
			if pc != 0x105c280 {
				b.Fatalf("want pc=0x105c280, got 0x%x", pc)
			}
			if f.Name != "main.main" {
				b.Fatalf("want name=main.main, got %q", f.Name)
			}
		}
	})

	b.Run("PCToLine", func { b ->
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			file, line, fn := tab.PCToLine(0x105c280)
			if file != "/tmp/hello.go" {
				b.Fatalf("want name=/tmp/hello.go, got %q", file)
			}
			if line != 3 {
				b.Fatalf("want line=3, got %d", line)
			}
			if fn.Name != "main.main" {
				b.Fatalf("want name=main.main, got %q", fn.Name)
			}
		}
	})
}
