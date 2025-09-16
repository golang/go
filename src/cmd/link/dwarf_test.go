// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	cmddwarf "cmd/internal/dwarf"
	"cmd/internal/objfile"
	"cmd/internal/quoted"
	"debug/dwarf"
	"internal/platform"
	"internal/testenv"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestMain allows this test binary to run as a -toolexec wrapper for
// the 'go' command. If LINK_TEST_TOOLEXEC is set, TestMain runs the
// binary as if it were cmd/link, and otherwise runs the requested
// tool as a subprocess.
//
// This allows the test to verify the behavior of the current contents of the
// cmd/link package even if the installed cmd/link binary is stale.
func TestMain(m *testing.M) {
	// Are we running as a toolexec wrapper? If so then run either
	// the correct tool or this executable itself (for the linker).
	// Running as toolexec wrapper.
	if os.Getenv("LINK_TEST_TOOLEXEC") != "" {
		if strings.TrimSuffix(filepath.Base(os.Args[1]), ".exe") == "link" {
			// Running as a -toolexec linker, and the tool is cmd/link.
			// Substitute this test binary for the linker.
			os.Args = os.Args[1:]
			main()
			os.Exit(0)
		}
		// Running some other tool.
		cmd := exec.Command(os.Args[1], os.Args[2:]...)
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			os.Exit(1)
		}
		os.Exit(0)
	}

	// Are we being asked to run as the linker (without toolexec)?
	// If so then kick off main.
	if os.Getenv("LINK_TEST_EXEC_LINKER") != "" {
		main()
		os.Exit(0)
	}

	if testExe, err := os.Executable(); err == nil {
		// on wasm, some phones, we expect an error from os.Executable()
		testLinker = testExe
	}

	// Not running as a -toolexec wrapper or as a linker executable.
	// Just run the tests.
	os.Exit(m.Run())
}

// Path of the test executable being run.
var testLinker string

func testDWARF(t *testing.T, buildmode string, expectDWARF bool, env ...string) {
	testenv.MustHaveCGO(t)
	testenv.MustHaveGoBuild(t)

	if !platform.ExecutableHasDWARF(runtime.GOOS, runtime.GOARCH) {
		t.Skipf("skipping on %s/%s: no DWARF symbol table in executables", runtime.GOOS, runtime.GOARCH)
	}

	t.Parallel()

	for _, prog := range []string{"testprog", "testprogcgo"} {
		prog := prog
		expectDWARF := expectDWARF
		if runtime.GOOS == "aix" && prog == "testprogcgo" {
			extld := os.Getenv("CC")
			if extld == "" {
				extld = "gcc"
			}
			extldArgs, err := quoted.Split(extld)
			if err != nil {
				t.Fatal(err)
			}
			expectDWARF, err = cmddwarf.IsDWARFEnabledOnAIXLd(extldArgs)
			if err != nil {
				t.Fatal(err)
			}
		}

		t.Run(prog, func(t *testing.T) {
			t.Parallel()

			tmpDir := t.TempDir()

			exe := filepath.Join(tmpDir, prog+".exe")
			dir := "../../runtime/testdata/" + prog
			cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-toolexec", os.Args[0], "-o", exe)
			if buildmode != "" {
				cmd.Args = append(cmd.Args, "-buildmode", buildmode)
			}
			cmd.Args = append(cmd.Args, dir)
			cmd.Env = append(os.Environ(), env...)
			cmd.Env = append(cmd.Env, "CGO_CFLAGS=") // ensure CGO_CFLAGS does not contain any flags. Issue #35459
			cmd.Env = append(cmd.Env, "LINK_TEST_TOOLEXEC=1")
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("go build -o %v %v: %v\n%s", exe, dir, err, out)
			}

			if buildmode == "c-archive" {
				// Extract the archive and use the go.o object within.
				ar := os.Getenv("AR")
				if ar == "" {
					ar = "ar"
				}
				cmd := testenv.Command(t, ar, "-x", exe)
				cmd.Dir = tmpDir
				if out, err := cmd.CombinedOutput(); err != nil {
					t.Fatalf("%s -x %s: %v\n%s", ar, exe, err, out)
				}
				exe = filepath.Join(tmpDir, "go.o")
			}

			darwinSymbolTestIsTooFlaky := true // Turn this off, it is too flaky -- See #32218
			if runtime.GOOS == "darwin" && !darwinSymbolTestIsTooFlaky {
				if _, err = exec.LookPath("symbols"); err == nil {
					// Ensure Apple's tooling can parse our object for symbols.
					out, err = testenv.Command(t, "symbols", exe).CombinedOutput()
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
		if !platform.BuildModeSupported(runtime.Compiler, "c-archive", runtime.GOOS, runtime.GOARCH) {
			t.Skipf("skipping c-archive test on unsupported platform %s-%s", runtime.GOOS, runtime.GOARCH)
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
	if err := testenv.Command(t, "xcrun", "--help").Run(); err != nil {
		t.Skipf("error running xcrun, required for iOS cross build: %v", err)
	}
	// Check to see if the ios tools are installed. It's possible to have the command line tools
	// installed without the iOS sdk.
	if output, err := testenv.Command(t, "xcodebuild", "-showsdks").CombinedOutput(); err != nil {
		t.Skipf("error running xcodebuild, required for iOS cross build: %v", err)
	} else if !strings.Contains(string(output), "iOS SDK") {
		t.Skipf("iOS SDK not detected.")
	}
	cc := "CC=" + runtime.GOROOT() + "/misc/ios/clangwrap.sh"
	// iOS doesn't allow unmapped segments, so iOS executables don't have DWARF.
	t.Run("exe", func(t *testing.T) {
		testDWARF(t, "", false, cc, "CGO_ENABLED=1", "GOOS=ios", "GOARCH=arm64")
	})
	// However, c-archive iOS objects have embedded DWARF.
	t.Run("c-archive", func(t *testing.T) {
		testDWARF(t, "c-archive", true, cc, "CGO_ENABLED=1", "GOOS=ios", "GOARCH=arm64")
	})
}

// This test ensures that variables promoted to the heap, specifically
// function return parameters, have correct location lists generated.
//
// TODO(deparker): This test is intentionally limited to GOOS=="linux"
// and scoped to net.sendFile, which was the function reported originally in
// issue #65405. There is relevant discussion in https://go-review.googlesource.com/c/go/+/684377
// pertaining to these limitations. There are other missing location lists which must be fixed
// particularly in functions where `linkname` is involved.
func TestDWARFLocationList(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("skipping test on non-linux OS")
	}
	testenv.MustHaveCGO(t)
	testenv.MustHaveGoBuild(t)

	if !platform.ExecutableHasDWARF(runtime.GOOS, runtime.GOARCH) {
		t.Skipf("skipping on %s/%s: no DWARF symbol table in executables", runtime.GOOS, runtime.GOARCH)
	}

	t.Parallel()

	tmpDir := t.TempDir()
	exe := filepath.Join(tmpDir, "issue65405.exe")
	dir := "./testdata/dwarf/issue65405"

	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-toolexec", os.Args[0], "-gcflags=all=-N -l", "-o", exe, dir)
	cmd.Env = append(os.Environ(), "CGO_CFLAGS=")
	cmd.Env = append(cmd.Env, "LINK_TEST_TOOLEXEC=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go build -o %v %v: %v\n%s", exe, dir, err, out)
	}

	f, err := objfile.Open(exe)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	d, err := f.DWARF()
	if err != nil {
		t.Fatal(err)
	}

	// Find the net.sendFile function and check its return parameter location list
	reader := d.Reader()

	for {
		entry, err := reader.Next()
		if err != nil {
			t.Fatal(err)
		}
		if entry == nil {
			break
		}

		// Look for the net.sendFile subprogram
		if entry.Tag == dwarf.TagSubprogram {
			fnName, ok := entry.Val(dwarf.AttrName).(string)
			if !ok || fnName != "net.sendFile" {
				reader.SkipChildren()
				continue
			}

			for {
				paramEntry, err := reader.Next()
				if err != nil {
					t.Fatal(err)
				}
				if paramEntry == nil || paramEntry.Tag == 0 {
					break
				}

				if paramEntry.Tag == dwarf.TagFormalParameter {
					paramName, _ := paramEntry.Val(dwarf.AttrName).(string)

					// Check if this parameter has a location attribute
					if loc := paramEntry.Val(dwarf.AttrLocation); loc != nil {
						switch locData := loc.(type) {
						case []byte:
							if len(locData) == 0 {
								t.Errorf("%s return parameter %q has empty location list", fnName, paramName)
								return
							}
						case int64:
							// Location list offset - this means it has a location list
							if locData == 0 {
								t.Errorf("%s return parameter %q has zero location list offset", fnName, paramName)
								return
							}
						default:
							t.Errorf("%s return parameter %q has unexpected location type %T: %v", fnName, paramName, locData, locData)
						}
					} else {
						t.Errorf("%s return parameter %q has no location attribute", fnName, paramName)
					}
				}
			}
		}
	}
}

func TestFlagW(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	tmpdir := t.TempDir()
	src := filepath.Join(tmpdir, "a.go")
	err := os.WriteFile(src, []byte(helloSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		flag      string
		wantDWARF bool
	}{
		{"-w", false},     // -w flag disables DWARF
		{"-s", false},     // -s implies -w
		{"-s -w=0", true}, // -w=0 negates the implied -w
	}
	for _, test := range tests {
		name := strings.ReplaceAll(test.flag, " ", "_")
		t.Run(name, func(t *testing.T) {
			ldflags := "-ldflags=" + test.flag
			exe := filepath.Join(t.TempDir(), "a.exe")
			cmd := testenv.Command(t, testenv.GoToolPath(t), "build", ldflags, "-o", exe, src)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("build failed: %v\n%s", err, out)
			}

			f, err := objfile.Open(exe)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()

			d, err := f.DWARF()
			if test.wantDWARF {
				if err != nil {
					t.Errorf("want binary with DWARF, got error %v", err)
				}
			} else {
				if d != nil {
					t.Errorf("want binary with no DWARF, got DWARF")
				}
			}
		})
	}
}
