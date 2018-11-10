// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/objfile"
	"debug/dwarf"
	"internal/testenv"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestDWARF(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("DWARF is not supported on Windows")
	}

	testenv.MustHaveCGO(t)
	testenv.MustHaveGoBuild(t)

	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; no DWARF symbol table in executables")
	}

	out, err := exec.Command(testenv.GoToolPath(t), "list", "-f", "{{.Stale}}", "cmd/link").CombinedOutput()
	if err != nil {
		t.Fatalf("go list: %v\n%s", err, out)
	}
	if string(out) != "false\n" {
		t.Fatalf("cmd/link is stale - run go install cmd/link")
	}

	tmpDir, err := ioutil.TempDir("", "go-link-TestDWARF")
	if err != nil {
		t.Fatal("TempDir failed: ", err)
	}
	defer os.RemoveAll(tmpDir)

	for _, prog := range []string{"testprog", "testprogcgo"} {
		t.Run(prog, func(t *testing.T) {
			exe := filepath.Join(tmpDir, prog+".exe")
			dir := "../../runtime/testdata/" + prog
			out, err := exec.Command(testenv.GoToolPath(t), "build", "-o", exe, dir).CombinedOutput()
			if err != nil {
				t.Fatalf("go build -o %v %v: %v\n%s", exe, dir, err, out)
			}

			f, err := objfile.Open(exe)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()

			syms, err := f.Symbols()
			if err != nil {
				t.Fatal(err)
			}

			var addr uint64
			for _, sym := range syms {
				if sym.Name == "main.main" {
					addr = sym.Addr
					break
				}
			}
			if addr == 0 {
				t.Fatal("cannot find main.main in symbols")
			}

			d, err := f.DWARF()
			if err != nil {
				t.Fatal(err)
			}

			// TODO: We'd like to use filepath.Join here.
			// Also related: golang.org/issue/19784.
			wantFile := path.Join(prog, "main.go")
			wantLine := 24
			r := d.Reader()
			var line dwarf.LineEntry
			for {
				cu, err := r.Next()
				if err != nil {
					t.Fatal(err)
				}
				if cu == nil {
					break
				}
				if cu.Tag != dwarf.TagCompileUnit {
					r.SkipChildren()
					continue
				}
				lr, err := d.LineReader(cu)
				if err != nil {
					t.Fatal(err)
				}
				for {
					err := lr.Next(&line)
					if err == io.EOF {
						break
					}
					if err != nil {
						t.Fatal(err)
					}
					if line.Address == addr {
						if !strings.HasSuffix(line.File.Name, wantFile) || line.Line != wantLine {
							t.Errorf("%#x is %s:%d, want %s:%d", addr, line.File.Name, line.Line, filepath.Join("...", wantFile), wantLine)
						}
						return
					}
				}
			}
			t.Fatalf("did not find file:line for %#x (main.main)", addr)
		})
	}
}
