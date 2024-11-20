// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa_test

import (
	cmddwarf "cmd/internal/dwarf"
	"cmd/internal/quoted"
	"cmp"
	"debug/dwarf"
	"debug/elf"
	"debug/macho"
	"debug/pe"
	"fmt"
	"internal/platform"
	"internal/testenv"
	"internal/xcoff"
	"io"
	"os"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func open(path string) (*dwarf.Data, error) {
	if fh, err := elf.Open(path); err == nil {
		return fh.DWARF()
	}

	if fh, err := pe.Open(path); err == nil {
		return fh.DWARF()
	}

	if fh, err := macho.Open(path); err == nil {
		return fh.DWARF()
	}

	if fh, err := xcoff.Open(path); err == nil {
		return fh.DWARF()
	}

	return nil, fmt.Errorf("unrecognized executable format")
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

type Line struct {
	File string
	Line int
}

func TestStmtLines(t *testing.T) {
	if !platform.ExecutableHasDWARF(runtime.GOOS, runtime.GOARCH) {
		t.Skipf("skipping on %s/%s: no DWARF symbol table in executables", runtime.GOOS, runtime.GOARCH)
	}

	if runtime.GOOS == "aix" {
		extld := os.Getenv("CC")
		if extld == "" {
			extld = "gcc"
		}
		extldArgs, err := quoted.Split(extld)
		if err != nil {
			t.Fatal(err)
		}
		enabled, err := cmddwarf.IsDWARFEnabledOnAIXLd(extldArgs)
		if err != nil {
			t.Fatal(err)
		}
		if !enabled {
			t.Skip("skipping on aix: no DWARF with ld version < 7.2.2 ")
		}
	}

	// Build cmd/go forcing DWARF enabled, as a large test case.
	dir := t.TempDir()
	out, err := testenv.Command(t, testenv.GoToolPath(t), "build", "-ldflags=-w=0", "-o", dir+"/test.exe", "cmd/go").CombinedOutput()
	if err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}

	lines := map[Line]bool{}
	dw, err := open(dir + "/test.exe")
	must(err)
	rdr := dw.Reader()
	rdr.Seek(0)
	for {
		e, err := rdr.Next()
		must(err)
		if e == nil {
			break
		}
		if e.Tag != dwarf.TagCompileUnit {
			continue
		}
		pkgname, _ := e.Val(dwarf.AttrName).(string)
		if pkgname == "runtime" {
			continue
		}
		if pkgname == "crypto/internal/fips/nistec/fiat" {
			continue // golang.org/issue/49372
		}
		if e.Val(dwarf.AttrStmtList) == nil {
			continue
		}
		lrdr, err := dw.LineReader(e)
		must(err)

		var le dwarf.LineEntry

		for {
			err := lrdr.Next(&le)
			if err == io.EOF {
				break
			}
			must(err)
			fl := Line{le.File.Name, le.Line}
			lines[fl] = lines[fl] || le.IsStmt
		}
	}

	nonStmtLines := []Line{}
	for line, isstmt := range lines {
		if !isstmt {
			nonStmtLines = append(nonStmtLines, line)
		}
	}

	var m int
	if runtime.GOARCH == "amd64" {
		m = 1 // > 99% obtained on amd64, no backsliding
	} else if runtime.GOARCH == "riscv64" {
		m = 3 // XXX temporary update threshold to 97% for regabi
	} else {
		m = 2 // expect 98% elsewhere.
	}

	if len(nonStmtLines)*100 > m*len(lines) {
		t.Errorf("Saw too many (%s, > %d%%) lines without statement marks, total=%d, nostmt=%d ('-run TestStmtLines -v' lists failing lines)\n", runtime.GOARCH, m, len(lines), len(nonStmtLines))
	}
	t.Logf("Saw %d out of %d lines without statement marks", len(nonStmtLines), len(lines))
	if testing.Verbose() {
		slices.SortFunc(nonStmtLines, func(a, b Line) int {
			if a.File != b.File {
				return strings.Compare(a.File, b.File)
			}
			return cmp.Compare(a.Line, b.Line)
		})
		for _, l := range nonStmtLines {
			t.Logf("%s:%d has no DWARF is_stmt mark\n", l.File, l.Line)
		}
	}
	t.Logf("total=%d, nostmt=%d\n", len(lines), len(nonStmtLines))
}
