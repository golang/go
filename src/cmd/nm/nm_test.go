// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/obscuretestdata"
	"internal/platform"
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"text/template"
)

// TestMain executes the test binary as the nm command if
// GO_NMTEST_IS_NM is set, and runs the tests otherwise.
func TestMain(m *testing.M) {
	if os.Getenv("GO_NMTEST_IS_NM") != "" {
		main()
		os.Exit(0)
	}

	os.Setenv("GO_NMTEST_IS_NM", "1") // Set for subprocesses to inherit.
	os.Exit(m.Run())
}

func TestNonGoExecs(t *testing.T) {
	t.Parallel()
	testfiles := []string{
		"debug/elf/testdata/gcc-386-freebsd-exec",
		"debug/elf/testdata/gcc-amd64-linux-exec",
		"debug/macho/testdata/gcc-386-darwin-exec.base64",   // golang.org/issue/34986
		"debug/macho/testdata/gcc-amd64-darwin-exec.base64", // golang.org/issue/34986
		// "debug/pe/testdata/gcc-amd64-mingw-exec", // no symbols!
		"debug/pe/testdata/gcc-386-mingw-exec",
		"debug/plan9obj/testdata/amd64-plan9-exec",
		"debug/plan9obj/testdata/386-plan9-exec",
		"internal/xcoff/testdata/gcc-ppc64-aix-dwarf2-exec",
	}
	for _, f := range testfiles {
		exepath := filepath.Join(testenv.GOROOT(t), "src", f)
		if strings.HasSuffix(f, ".base64") {
			tf, err := obscuretestdata.DecodeToTempFile(exepath)
			if err != nil {
				t.Errorf("obscuretestdata.DecodeToTempFile(%s): %v", exepath, err)
				continue
			}
			defer os.Remove(tf)
			exepath = tf
		}

		cmd := testenv.Command(t, testenv.Executable(t), exepath)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("go tool nm %v: %v\n%s", exepath, err, string(out))
		}
	}
}

func testGoExec(t *testing.T, iscgo, isexternallinker bool) {
	t.Parallel()
	tmpdir, err := os.MkdirTemp("", "TestGoExec")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	src := filepath.Join(tmpdir, "a.go")
	file, err := os.Create(src)
	if err != nil {
		t.Fatal(err)
	}
	err = template.Must(template.New("main").Parse(testexec)).Execute(file, iscgo)
	if e := file.Close(); err == nil {
		err = e
	}
	if err != nil {
		t.Fatal(err)
	}

	exe := filepath.Join(tmpdir, "a.exe")
	args := []string{"build", "-o", exe}
	if iscgo {
		linkmode := "internal"
		if isexternallinker {
			linkmode = "external"
		}
		args = append(args, "-ldflags", "-linkmode="+linkmode)
	}
	args = append(args, src)
	out, err := testenv.Command(t, testenv.GoToolPath(t), args...).CombinedOutput()
	if err != nil {
		t.Fatalf("building test executable failed: %s %s", err, out)
	}

	out, err = testenv.Command(t, exe).CombinedOutput()
	if err != nil {
		t.Fatalf("running test executable failed: %s %s", err, out)
	}
	names := make(map[string]string)
	for _, line := range strings.Split(string(out), "\n") {
		if line == "" {
			continue
		}
		f := strings.Split(line, "=")
		if len(f) != 2 {
			t.Fatalf("unexpected output line: %q", line)
		}
		names["main."+f[0]] = f[1]
	}

	runtimeSyms := map[string]string{
		"runtime.text":      "T",
		"runtime.etext":     "T",
		"runtime.rodata":    "R",
		"runtime.erodata":   "R",
		"runtime.epclntab":  "R",
		"runtime.noptrdata": "D",
	}

	if runtime.GOOS == "aix" && iscgo {
		// pclntab is moved to .data section on AIX.
		runtimeSyms["runtime.epclntab"] = "D"
	}

	out, err = testenv.Command(t, testenv.Executable(t), exe).CombinedOutput()
	if err != nil {
		t.Fatalf("go tool nm: %v\n%s", err, string(out))
	}

	relocated := func(code string) bool {
		if runtime.GOOS == "aix" {
			// On AIX, .data and .bss addresses are changed by the loader.
			// Therefore, the values returned by the exec aren't the same
			// than the ones inside the symbol table.
			// In case of cgo, .text symbols are also changed.
			switch code {
			case "T", "t", "R", "r":
				return iscgo
			case "D", "d", "B", "b":
				return true
			}
		}
		if platform.DefaultPIE(runtime.GOOS, runtime.GOARCH, false) {
			// Code is always relocated if the default buildmode is PIE.
			return true
		}
		return false
	}

	dups := make(map[string]bool)
	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Fields(line)
		if len(f) < 3 {
			continue
		}
		name := f[2]
		if addr, found := names[name]; found {
			if want, have := addr, "0x"+f[0]; have != want {
				if !relocated(f[1]) {
					t.Errorf("want %s address for %s symbol, but have %s", want, name, have)
				}
			}
			delete(names, name)
		}
		if _, found := dups[name]; found {
			t.Errorf("duplicate name of %q is found", name)
		}
		if stype, found := runtimeSyms[name]; found {
			if runtime.GOOS == "plan9" && stype == "R" {
				// no read-only data segment symbol on Plan 9
				stype = "D"
			}
			if want, have := stype, strings.ToUpper(f[1]); have != want {
				if runtime.GOOS == "android" && name == "runtime.epclntab" && have == "D" {
					// TODO(#58807): Figure out why this fails and fix up the test.
					t.Logf("(ignoring on %s) want %s type for %s symbol, but have %s", runtime.GOOS, want, name, have)
				} else {
					t.Errorf("want %s type for %s symbol, but have %s", want, name, have)
				}
			}
			delete(runtimeSyms, name)
		}
	}
	if len(names) > 0 {
		t.Errorf("executable is missing %v symbols", names)
	}
	if len(runtimeSyms) > 0 {
		t.Errorf("executable is missing %v symbols", runtimeSyms)
	}
}

func TestGoExec(t *testing.T) {
	testGoExec(t, false, false)
}

func testGoLib(t *testing.T, iscgo bool) {
	t.Parallel()
	tmpdir, err := os.MkdirTemp("", "TestGoLib")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	gopath := filepath.Join(tmpdir, "gopath")
	libpath := filepath.Join(gopath, "src", "mylib")

	err = os.MkdirAll(libpath, 0777)
	if err != nil {
		t.Fatal(err)
	}
	src := filepath.Join(libpath, "a.go")
	file, err := os.Create(src)
	if err != nil {
		t.Fatal(err)
	}
	err = template.Must(template.New("mylib").Parse(testlib)).Execute(file, iscgo)
	if e := file.Close(); err == nil {
		err = e
	}
	if err == nil {
		err = os.WriteFile(filepath.Join(libpath, "go.mod"), []byte("module mylib\n"), 0666)
	}
	if err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-buildmode=archive", "-o", "mylib.a", ".")
	cmd.Dir = libpath
	cmd.Env = append(os.Environ(), "GOPATH="+gopath)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("building test lib failed: %s %s", err, out)
	}
	mylib := filepath.Join(libpath, "mylib.a")

	out, err = testenv.Command(t, testenv.Executable(t), mylib).CombinedOutput()
	if err != nil {
		t.Fatalf("go tool nm: %v\n%s", err, string(out))
	}
	type symType struct {
		Type  string
		Name  string
		CSym  bool
		Found bool
	}
	var syms = []symType{
		{"B", "mylib.Testdata", false, false},
		{"T", "mylib.Testfunc", false, false},
	}
	if iscgo {
		syms = append(syms, symType{"B", "mylib.TestCgodata", false, false})
		syms = append(syms, symType{"T", "mylib.TestCgofunc", false, false})
		if runtime.GOOS == "darwin" || runtime.GOOS == "ios" || (runtime.GOOS == "windows" && runtime.GOARCH == "386") {
			syms = append(syms, symType{"D", "_cgodata", true, false})
			syms = append(syms, symType{"T", "_cgofunc", true, false})
		} else if runtime.GOOS == "aix" {
			syms = append(syms, symType{"D", "cgodata", true, false})
			syms = append(syms, symType{"T", ".cgofunc", true, false})
		} else {
			syms = append(syms, symType{"D", "cgodata", true, false})
			syms = append(syms, symType{"T", "cgofunc", true, false})
		}
	}

	for _, line := range strings.Split(string(out), "\n") {
		f := strings.Fields(line)
		var typ, name string
		var csym bool
		if iscgo {
			if len(f) < 4 {
				continue
			}
			csym = !strings.Contains(f[0], "_go_.o")
			typ = f[2]
			name = f[3]
		} else {
			if len(f) < 3 {
				continue
			}
			typ = f[1]
			name = f[2]
		}
		for i := range syms {
			sym := &syms[i]
			if sym.Type == typ && sym.Name == name && sym.CSym == csym {
				if sym.Found {
					t.Fatalf("duplicate symbol %s %s", sym.Type, sym.Name)
				}
				sym.Found = true
			}
		}
	}
	for _, sym := range syms {
		if !sym.Found {
			t.Errorf("cannot found symbol %s %s", sym.Type, sym.Name)
		}
	}
}

func TestGoLib(t *testing.T) {
	testGoLib(t, false)
}

const testexec = `
package main

import "fmt"
{{if .}}import "C"
{{end}}

func main() {
	testfunc()
}

var testdata uint32

func testfunc() {
	fmt.Printf("main=%p\n", main)
	fmt.Printf("testfunc=%p\n", testfunc)
	fmt.Printf("testdata=%p\n", &testdata)
}
`

const testlib = `
package mylib

{{if .}}
// int cgodata = 5;
// void cgofunc(void) {}
import "C"

var TestCgodata = C.cgodata

func TestCgofunc() {
	C.cgofunc()
}
{{end}}

var Testdata uint32

func Testfunc() {}
`
