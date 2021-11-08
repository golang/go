// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"crypto/md5"
	"flag"
	"fmt"
	"go/build"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

var tmp, exe string // populated by buildObjdump

func TestMain(m *testing.M) {
	if !testenv.HasGoBuild() {
		return
	}

	var exitcode int
	if err := buildObjdump(); err == nil {
		exitcode = m.Run()
	} else {
		fmt.Println(err)
		exitcode = 1
	}
	os.RemoveAll(tmp)
	os.Exit(exitcode)
}

func buildObjdump() error {
	var err error
	tmp, err = os.MkdirTemp("", "TestObjDump")
	if err != nil {
		return fmt.Errorf("TempDir failed: %v", err)
	}

	exe = filepath.Join(tmp, "testobjdump.exe")
	gotool, err := testenv.GoTool()
	if err != nil {
		return err
	}
	out, err := exec.Command(gotool, "build", "-o", exe, "cmd/objdump").CombinedOutput()
	if err != nil {
		os.RemoveAll(tmp)
		return fmt.Errorf("go build -o %v cmd/objdump: %v\n%s", exe, err, string(out))
	}

	return nil
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

var ppcNeed = []string{
	"BR main.main(SB)",
	"CALL main.Println(SB)",
	"RET",
}

var ppcGnuNeed = []string{
	"mflr",
	"lbz",
	"cmpw",
}

func mustHaveDisasm(t *testing.T) {
	switch runtime.GOARCH {
	case "mips", "mipsle", "mips64", "mips64le":
		t.Skipf("skipping on %s, issue 12559", runtime.GOARCH)
	case "riscv64":
		t.Skipf("skipping on %s, issue 36738", runtime.GOARCH)
	case "s390x":
		t.Skipf("skipping on %s, issue 15255", runtime.GOARCH)
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

	hash := md5.Sum([]byte(fmt.Sprintf("%v-%v-%v-%v", srcfname, flags, printCode, printGnuAsm)))
	hello := filepath.Join(tmp, fmt.Sprintf("hello-%x.exe", hash))
	args := []string{"build", "-o", hello}
	args = append(args, flags...)
	args = append(args, srcfname)
	cmd := exec.Command(testenv.GoToolPath(t), args...)
	// "Bad line" bug #36683 is sensitive to being run in the source directory.
	cmd.Dir = "testdata"
	// Ensure that the source file location embedded in the binary matches our
	// actual current GOROOT, instead of GOROOT_FINAL if set.
	cmd.Env = append(os.Environ(), "GOROOT_FINAL=")
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
	case "ppc64", "ppc64le":
		need = append(need, ppcNeed...)
	}

	if printGnuAsm {
		switch goarch {
		case "amd64":
			need = append(need, amd64GnuNeed...)
		case "386":
			need = append(need, i386GnuNeed...)
		case "arm", "arm64":
			need = append(need, armGnuNeed...)
		case "ppc64", "ppc64le":
			need = append(need, ppcGnuNeed...)
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
	cmd = exec.Command(exe, args...)
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
	if build.Default.CgoEnabled {
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
	case "plan9", "windows":
		t.Skipf("skipping on %s", runtime.GOOS)
	}
	t.Parallel()
	testDisasm(t, "fmthello.go", false, false, "-ldflags=-linkmode=external")
}

func TestDisasmGoobj(t *testing.T) {
	mustHaveDisasm(t)

	hello := filepath.Join(tmp, "hello.o")
	args := []string{"tool", "compile", "-o", hello}
	args = append(args, "testdata/fmthello.go")
	out, err := exec.Command(testenv.GoToolPath(t), args...).CombinedOutput()
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

	out, err = exec.Command(exe, args...).CombinedOutput()
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

	tmpdir, err := os.MkdirTemp("", "TestGoobjFileNumber")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	obj := filepath.Join(tmpdir, "p.a")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", obj)
	cmd.Dir = filepath.Join("testdata/testfilenum")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build failed: %v\n%s", err, out)
	}

	cmd = exec.Command(exe, obj)
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
	testenv.MustHaveExec(t)
	t.Parallel()

	obj := filepath.Join("testdata", "go116.o")
	cmd := exec.Command(exe, obj)
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("objdump go116.o succeeded unexpectly")
	}
	if !strings.Contains(string(out), "go object of a different version") {
		t.Errorf("unexpected error message:\n%s", out)
	}
}
