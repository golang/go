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

func testDWARF(t *testing.T, buildmode string, expectDWARF bool, env ...string) {
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
		if os.Getenv("GOROOT_FINAL_OLD") != "" {
			t.Skip("cmd/link is stale, but $GOROOT_FINAL_OLD is set")
		}
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
			cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", exe)
			if buildmode != "" {
				cmd.Args = append(cmd.Args, "-buildmode", buildmode)
			}
			cmd.Args = append(cmd.Args, dir)
			if env != nil {
				cmd.Env = append(os.Environ(), env...)
			}
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("go build -o %v %v: %v\n%s", exe, dir, err, out)
			}

			if buildmode == "c-archive" {
				// Extract the archive and use the go.o object within.
				cmd := exec.Command("ar", "-x", exe)
				cmd.Dir = tmpDir
				if out, err := cmd.CombinedOutput(); err != nil {
					t.Fatalf("ar -x %s: %v\n%s", exe, err, out)
				}
				exe = filepath.Join(tmpDir, "go.o")
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
				if expectDWARF {
					t.Fatal(err)
				}
				return
			} else {
				if !expectDWARF {
					t.Fatal("unexpected DWARF section")
				}
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

func TestDWARF(t *testing.T) {
	testDWARF(t, "", true)
}

func TestDWARFiOS(t *testing.T) {
	// Normally we run TestDWARF on native platform. But on iOS we don't have
	// go build, so we do this test with a cross build.
	// Only run this on darwin/amd64, where we can cross build for iOS.
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	if runtime.GOARCH != "amd64" || runtime.GOOS != "darwin" {
		t.Skip("skipping on non-darwin/amd64 platform")
	}
	if err := exec.Command("xcrun", "--help").Run(); err != nil {
		t.Skipf("error running xcrun, required for iOS cross build: %v", err)
	}
	cc := "CC=" + runtime.GOROOT() + "/misc/ios/clangwrap.sh"
	// iOS doesn't allow unmapped segments, so iOS executables don't have DWARF.
	testDWARF(t, "", false, cc, "CGO_ENABLED=1", "GOOS=darwin", "GOARCH=arm", "GOARM=7")
	testDWARF(t, "", false, cc, "CGO_ENABLED=1", "GOOS=darwin", "GOARCH=arm64")
	// However, c-archive iOS objects have embedded DWARF.
	testDWARF(t, "c-archive", true, cc, "CGO_ENABLED=1", "GOOS=darwin", "GOARCH=arm", "GOARM=7")
	testDWARF(t, "c-archive", true, cc, "CGO_ENABLED=1", "GOOS=darwin", "GOARCH=arm64")
}
