// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/internal/hash"
	"flag"
	"fmt"
	"internal/platform"
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestMain executes the test binary as the objdump command if
// GO_OBJDUMPTEST_IS_OBJDUMP is set, and runs the test otherwise.
func TestMain(m *testing.M) {
	if os.Getenv("GO_OBJDUMPTEST_IS_OBJDUMP") != "" {
		main()
		os.Exit(0)
	}

	os.Setenv("GO_OBJDUMPTEST_IS_OBJDUMP", "1")
	os.Exit(m.Run())
}

var x86Need = []string{ // for both 386 and AMD64
	"JMP main.main(SB)",
	"CALL main.Println(SB)",
	"RET",
}

var amd64GnuNeed = []string{
	"jmp",
	"callq",
	"cmpb",
}

var i386GnuNeed = []string{
	"jmp",
	"call",
	"cmp",
}

var armNeed = []string{
	"B main.main(SB)",
	"BL main.Println(SB)",
	"RET",
}

var arm64Need = []string{
	"JMP main.main(SB)",
	"CALL main.Println(SB)",
	"RET",
}

var armGnuNeed = []string{ // for both ARM and AMR64
	"ldr",
	"bl",
	"cmp",
}

var loong64Need = []string{
	"JMP main.main(SB)",
	"CALL main.Println(SB)",
	"RET",
}

var loong64GnuNeed = []string{
	"ld.b",
	"bl",
	"beq",
}

var ppcNeed = []string{
	"BR main.main(SB)",
	"CALL main.Println(SB)",
	"RET",
}

var ppcPIENeed = []string{
	"BR",
	"CALL",
	"RET",
}

var ppcGnuNeed = []string{
	"mflr",
	"lbz",
	"beq",
}

var s390xGnuNeed = []string{
	"brasl",
	"j",
	"clije",
}

func mustHaveDisasm(t *testing.T) {
	switch runtime.GOARCH {
	case "mips", "mipsle", "mips64", "mips64le":
		t.Skipf("skipping on %s, issue 12559", runtime.GOARCH)
	}
}

var target = flag.String("target", "", "test disassembly of `goos/goarch` binary")

// objdump is fully cross platform: it can handle binaries
// from any known operating system and architecture.
// We could in principle add binaries to testdata and check
// all the supported systems during this test. However, the
// binaries would be about 1 MB each, and we don't want to
// add that much junk to the hg repository. Instead, build a
// binary for the current system (only) and test that objdump
// can handle that one.

func testDisasm(t *testing.T, srcfname string, printCode bool, printGnuAsm bool, flags ...string) {
	mustHaveDisasm(t)
	goarch := runtime.GOARCH
	if *target != "" {
		f := strings.Split(*target, "/")
		if len(f) != 2 {
			t.Fatalf("-target argument must be goos/goarch")
		}
		defer os.Setenv("GOOS", os.Getenv("GOOS"))
		defer os.Setenv("GOARCH", os.Getenv("GOARCH"))
		os.Setenv("GOOS", f[0])
		os.Setenv("GOARCH", f[1])
		goarch = f[1]
	}

	hash := hash.Sum16([]byte(fmt.Sprintf("%v-%v-%v-%v", srcfname, flags, printCode, printGnuAsm)))
	tmp := t.TempDir()
	hello := filepath.Join(tmp, fmt.Sprintf("hello-%x.exe", hash))
	args := []string{"build", "-o", hello}
	args = append(args, flags...)
	args = append(args, srcfname)
	cmd := testenv.Command(t, testenv.GoToolPath(t), args...)
	// "Bad line" bug #36683 is sensitive to being run in the source directory.
	cmd.Dir = "testdata"
	t.Logf("Running %v", cmd.Args)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go build %s: %v\n%s", srcfname, err, out)
	}
	need := []string{
		"TEXT main.main(SB)",
	}

	if printCode {
		need = append(need, `	Println("hello, world")`)
	} else {
		need = append(need, srcfname+":6")
	}

	switch goarch {
	case "amd64", "386":
		need = append(need, x86Need...)
	case "arm":
		need = append(need, armNeed...)
	case "arm64":
		need = append(need, arm64Need...)
	case "loong64":
		need = append(need, loong64Need...)
	case "ppc64", "ppc64le":
		var pie bool
		for _, flag := range flags {
			if flag == "-buildmode=pie" {
				pie = true
				break
			}
		}
		if pie {
			// In PPC64 PIE binaries we use a "local entry point" which is
			// function symbol address + 8. Currently we don't symbolize that.
			// Expect a different output.
			need = append(need, ppcPIENeed...)
		} else {
			need = append(need, ppcNeed...)
		}
	}

	if printGnuAsm {
		switch goarch {
		case "amd64":
			need = append(need, amd64GnuNeed...)
		case "386":
			need = append(need, i386GnuNeed...)
		case "arm", "arm64":
			need = append(need, armGnuNeed...)
		case "loong64":
			need = append(need, loong64GnuNeed...)
		case "ppc64", "ppc64le":
			need = append(need, ppcGnuNeed...)
		case "s390x":
			need = append(need, s390xGnuNeed...)
		}
	}
	args = []string{
		"-s", "main.main",
		hello,
	}

	if printCode {
		args = append([]string{"-S"}, args...)
	}

	if printGnuAsm {
		args = append([]string{"-gnu"}, args...)
	}
	cmd = testenv.Command(t, testenv.Executable(t), args...)
	cmd.Dir = "testdata" // "Bad line" bug #36683 is sensitive to being run in the source directory
	out, err = cmd.CombinedOutput()
	t.Logf("Running %v", cmd.Args)

	if err != nil {
		exename := srcfname[:len(srcfname)-len(filepath.Ext(srcfname))] + ".exe"
		t.Fatalf("objdump %q: %v\n%s", exename, err, out)
	}

	text := string(out)
	ok := true
	for _, s := range need {
		if !strings.Contains(text, s) {
			t.Errorf("disassembly missing '%s'", s)
			ok = false
		}
	}
	if goarch == "386" {
		if strings.Contains(text, "(IP)") {
			t.Errorf("disassembly contains PC-Relative addressing on 386")
			ok = false
		}
	}

	if !ok || testing.Verbose() {
		t.Logf("full disassembly:\n%s", text)
	}
}

func testGoAndCgoDisasm(t *testing.T, printCode bool, printGnuAsm bool) {
	t.Parallel()
	testDisasm(t, "fmthello.go", printCode, printGnuAsm)
	if testenv.HasCGO() {
		testDisasm(t, "fmthellocgo.go", printCode, printGnuAsm)
	}
}

func TestDisasm(t *testing.T) {
	testGoAndCgoDisasm(t, false, false)
}

func TestDisasmCode(t *testing.T) {
	testGoAndCgoDisasm(t, true, false)
}

func TestDisasmGnuAsm(t *testing.T) {
	testGoAndCgoDisasm(t, false, true)
}

func TestDisasmExtld(t *testing.T) {
	testenv.MustHaveCGO(t)
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping on %s", runtime.GOOS)
	}
	t.Parallel()
	testDisasm(t, "fmthello.go", false, false, "-ldflags=-linkmode=external")
}

func TestDisasmPIE(t *testing.T) {
	if !platform.BuildModeSupported("gc", "pie", runtime.GOOS, runtime.GOARCH) {
		t.Skipf("skipping on %s/%s, PIE buildmode not supported", runtime.GOOS, runtime.GOARCH)
	}
	if !platform.InternalLinkPIESupported(runtime.GOOS, runtime.GOARCH) {
		// require cgo on platforms that PIE needs external linking
		testenv.MustHaveCGO(t)
	}
	t.Parallel()
	testDisasm(t, "fmthello.go", false, false, "-buildmode=pie")
}

func TestDisasmGoobj(t *testing.T) {
	mustHaveDisasm(t)
	testenv.MustHaveGoBuild(t)

	tmp := t.TempDir()

	importcfgfile := filepath.Join(tmp, "hello.importcfg")
	testenv.WriteImportcfg(t, importcfgfile, nil, "testdata/fmthello.go")

	hello := filepath.Join(tmp, "hello.o")
	args := []string{"tool", "compile", "-p=main", "-importcfg=" + importcfgfile, "-o", hello}
	args = append(args, "testdata/fmthello.go")
	out, err := testenv.Command(t, testenv.GoToolPath(t), args...).CombinedOutput()
	if err != nil {
		t.Fatalf("go tool compile fmthello.go: %v\n%s", err, out)
	}
	need := []string{
		"main(SB)",
		"fmthello.go:6",
	}

	args = []string{
		"-s", "main",
		hello,
	}

	out, err = testenv.Command(t, testenv.Executable(t), args...).CombinedOutput()
	if err != nil {
		t.Fatalf("objdump fmthello.o: %v\n%s", err, out)
	}

	text := string(out)
	ok := true
	for _, s := range need {
		if !strings.Contains(text, s) {
			t.Errorf("disassembly missing '%s'", s)
			ok = false
		}
	}
	if runtime.GOARCH == "386" {
		if strings.Contains(text, "(IP)") {
			t.Errorf("disassembly contains PC-Relative addressing on 386")
			ok = false
		}
	}
	if !ok {
		t.Logf("full disassembly:\n%s", text)
	}
}

func TestGoobjFileNumber(t *testing.T) {
	// Test that file table in Go object file is parsed correctly.
	testenv.MustHaveGoBuild(t)
	mustHaveDisasm(t)

	t.Parallel()

	tmp := t.TempDir()

	obj := filepath.Join(tmp, "p.a")
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", obj)
	cmd.Dir = filepath.Join("testdata/testfilenum")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build failed: %v\n%s", err, out)
	}

	cmd = testenv.Command(t, testenv.Executable(t), obj)
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("objdump failed: %v\n%s", err, out)
	}

	text := string(out)
	for _, s := range []string{"a.go", "b.go", "c.go"} {
		if !strings.Contains(text, s) {
			t.Errorf("output missing '%s'", s)
		}
	}

	if t.Failed() {
		t.Logf("output:\n%s", text)
	}
}

func TestGoObjOtherVersion(t *testing.T) {
	t.Parallel()

	obj := filepath.Join("testdata", "go116.o")
	cmd := testenv.Command(t, testenv.Executable(t), obj)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("objdump go116.o succeeded unexpectedly")
	}
	if !strings.Contains(string(out), "go object of a different version") {
		t.Errorf("unexpected error message:\n%s", out)
	}
}
