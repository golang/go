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
			cmd := goCmd(t, "build", "-o", exe)
			if buildmode != "" {
				cmd.Args = append(cmd.Args, "-buildmode", buildmode)
			}
			cmd.Args = append(cmd.Args, dir)
			cmd.Env = append(cmd.Env, env...)
			cmd.Env = append(cmd.Env, "CGO_CFLAGS=") // ensure CGO_CFLAGS does not contain any flags. Issue #35459
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

			if buildmode != "c-archive" {
				testModuledata(t, d)
			}
		})
	}
}

// testModuledata makes sure that runtime.firstmoduledata exists
// and has a type. Issue #76731.
func testModuledata(t *testing.T, d *dwarf.Data) {
	const symName = "runtime.firstmoduledata"

	r := d.Reader()
	for {
		e, err := r.Next()
		if err != nil {
			t.Error(err)
			return
		}
		if e == nil {
			t.Errorf("did not find DWARF entry for %s", symName)
			return
		}

		switch e.Tag {
		case dwarf.TagVariable:
			// carry on after switch
		case dwarf.TagCompileUnit, dwarf.TagSubprogram:
			continue
		default:
			r.SkipChildren()
			continue
		}

		nameIdx, typeIdx := -1, -1
		for i := range e.Field {
			f := &e.Field[i]
			switch f.Attr {
			case dwarf.AttrName:
				nameIdx = i
			case dwarf.AttrType:
				typeIdx = i
			}
		}
		if nameIdx == -1 {
			// unnamed variable?
			r.SkipChildren()
			continue
		}
		nameStr, ok := e.Field[nameIdx].Val.(string)
		if !ok {
			// variable name is not a string?
			r.SkipChildren()
			continue
		}
		if nameStr != symName {
			r.SkipChildren()
			continue
		}

		if typeIdx == -1 {
			t.Errorf("%s has no DWARF type", symName)
			return
		}
		off, ok := e.Field[typeIdx].Val.(dwarf.Offset)
		if !ok {
			t.Errorf("unexpected Go type %T for DWARF type for %s; expected %T", e.Field[typeIdx].Val, symName, dwarf.Offset(0))
			return
		}

		typeInfo, err := d.Type(off)
		if err != nil {
			t.Error(err)
			return
		}

		typeName := typeInfo.Common().Name
		if want := "runtime.moduledata"; typeName != want {
			t.Errorf("type of %s is %s, expected %s", symName, typeName, want)
		}
		for {
			typedef, ok := typeInfo.(*dwarf.TypedefType)
			if !ok {
				break
			}
			typeInfo = typedef.Type
		}
		if _, ok := typeInfo.(*dwarf.StructType); !ok {
			t.Errorf("type of %s is %T, expected %T", symName, typeInfo, dwarf.StructType{})
		}

		return
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

	cmd := goCmd(t, "build", "-gcflags=all=-N -l", "-o", exe, dir)
	cmd.Env = append(cmd.Env, "CGO_CFLAGS=")
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
	if runtime.GOOS == "aix" {
		t.Skip("internal/xcoff cannot parse file without symbol table")
	}
	if !platform.ExecutableHasDWARF(runtime.GOOS, runtime.GOARCH) {
		t.Skipf("skipping on %s/%s: no DWARF symbol table in executables", runtime.GOOS, runtime.GOARCH)
	}

	t.Parallel()

	tmpdir := t.TempDir()
	src := filepath.Join(tmpdir, "a.go")
	err := os.WriteFile(src, []byte(helloSrc), 0666)
	if err != nil {
		t.Fatal(err)
	}

	type testCase struct {
		flag      string
		wantDWARF bool
	}
	tests := []testCase{
		{"-w", false},     // -w flag disables DWARF
		{"-s", false},     // -s implies -w
		{"-s -w=0", true}, // -w=0 negates the implied -w
	}
	if testenv.HasCGO() && runtime.GOOS != "solaris" { // Solaris linker doesn't support the -S flag
		tests = append(tests,
			testCase{"-w -linkmode=external", false},
			testCase{"-s -linkmode=external", false},
			// Some external linkers don't have a way to preserve DWARF
			// without emitting the symbol table. Skip this case for now.
			// I suppose we can post- process, e.g. with objcopy.
			//testCase{"-s -w=0 -linkmode=external", true},
		)
	}

	for _, test := range tests {
		name := strings.ReplaceAll(test.flag, " ", "_")
		t.Run(name, func(t *testing.T) {
			ldflags := "-ldflags=" + test.flag
			exe := filepath.Join(t.TempDir(), "a.exe")
			cmd := goCmd(t, "build", ldflags, "-o", exe, src)
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
