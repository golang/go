// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	cmddwarf "cmd/internal/dwarf"
	"cmd/internal/objfile"
	"debug/dwarf"
	"internal/testenv"
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
		if strings.HasPrefix(testenv.Builder(), "darwin-") {
			t.Skipf("cmd/link is spuriously stale on Darwin builders - see #33598")
		}
		t.Fatalf("cmd/link is stale - run go install cmd/link")
	}

	for _, prog := range []string{"testprog", "testprogcgo"} {
		prog := prog
		expectDWARF := expectDWARF
		if runtime.GOOS == "aix" && prog == "testprogcgo" {
			extld := os.Getenv("CC")
			if extld == "" {
				extld = "gcc"
			}
			expectDWARF, err = cmddwarf.IsDWARFEnabledOnAIXLd(extld)
			if err != nil {
				t.Fatal(err)
			}

		}

		t.Run(prog, func(t *testing.T) {
			t.Parallel()

			tmpDir, err := ioutil.TempDir("", "go-link-TestDWARF")
			if err != nil {
				t.Fatal(err)
			}
			defer os.RemoveAll(tmpDir)

			exe := filepath.Join(tmpDir, prog+".exe")
			dir := "../../runtime/testdata/" + prog
			cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", exe)
			if buildmode != "" {
				cmd.Args = append(cmd.Args, "-buildmode", buildmode)
			}
			cmd.Args = append(cmd.Args, dir)
			if env != nil {
				cmd.Env = append(os.Environ(), env...)
				cmd.Env = append(cmd.Env, "CGO_CFLAGS=") // ensure CGO_CFLAGS does not contain any flags. Issue #35459
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

			if runtime.GOOS == "darwin" {
				if _, err = exec.LookPath("symbols"); err == nil {
					// Ensure Apple's tooling can parse our object for symbols.
					out, err = exec.Command("symbols", exe).CombinedOutput()
					if err != nil {
						t.Fatalf("symbols %v: %v: %s", filepath.Base(exe), err, out)
					} else {
						if bytes.HasPrefix(out, []byte("Unable to find file")) {
							// This failure will cause the App Store to reject our binaries.
							t.Fatalf("symbols %v: failed to parse file", filepath.Base(exe))
						} else if bytes.Contains(out, []byte(", Empty]")) {
							t.Fatalf("symbols %v: parsed as empty", filepath.Base(exe))
						}
					}
				}
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
			entry, err := r.SeekPC(addr)
			if err != nil {
				t.Fatal(err)
			}
			lr, err := d.LineReader(entry)
			if err != nil {
				t.Fatal(err)
			}
			var line dwarf.LineEntry
			if err := lr.SeekPC(addr, &line); err == dwarf.ErrUnknownPC {
				t.Fatalf("did not find file:line for %#x (main.main)", addr)
			} else if err != nil {
				t.Fatal(err)
			}
			if !strings.HasSuffix(line.File.Name, wantFile) || line.Line != wantLine {
				t.Errorf("%#x is %s:%d, want %s:%d", addr, line.File.Name, line.Line, filepath.Join("...", wantFile), wantLine)
			}
		})
	}
}

func TestDWARF(t *testing.T) {
	testDWARF(t, "", true)
	if !testing.Short() {
		if runtime.GOOS == "windows" {
			t.Skip("skipping Windows/c-archive; see Issue 35512 for more.")
		}
		t.Run("c-archive", func(t *testing.T) {
			testDWARF(t, "c-archive", true)
		})
	}
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
	testDWARF(t, "", false, cc, "CGO_ENABLED=1", "GOOS=darwin", "GOARCH=arm64")
	// However, c-archive iOS objects have embedded DWARF.
	testDWARF(t, "c-archive", true, cc, "CGO_ENABLED=1", "GOOS=darwin", "GOARCH=arm64")
}
