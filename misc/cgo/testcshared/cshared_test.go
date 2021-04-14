// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cshared_test

import (
	"bytes"
	"debug/elf"
	"debug/pe"
	"encoding/binary"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"unicode"
)

// C compiler with args (from $(go env CC) $(go env GOGCCFLAGS)).
var cc []string

// ".exe" on Windows.
var exeSuffix string

var GOOS, GOARCH, GOROOT string
var installdir, androiddir string
var libSuffix, libgoname string

func TestMain(m *testing.M) {
	os.Exit(testMain(m))
}

func testMain(m *testing.M) int {
	log.SetFlags(log.Lshortfile)
	flag.Parse()
	if testing.Short() && os.Getenv("GO_BUILDER_NAME") == "" {
		fmt.Printf("SKIP - short mode and $GO_BUILDER_NAME not set\n")
		os.Exit(0)
	}

	GOOS = goEnv("GOOS")
	GOARCH = goEnv("GOARCH")
	GOROOT = goEnv("GOROOT")

	if _, err := os.Stat(GOROOT); os.IsNotExist(err) {
		log.Fatalf("Unable able to find GOROOT at '%s'", GOROOT)
	}

	androiddir = fmt.Sprintf("/data/local/tmp/testcshared-%d", os.Getpid())
	if runtime.GOOS != GOOS && GOOS == "android" {
		args := append(adbCmd(), "exec-out", "mkdir", "-p", androiddir)
		cmd := exec.Command(args[0], args[1:]...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			log.Fatalf("setupAndroid failed: %v\n%s\n", err, out)
		}
		defer cleanupAndroid()
	}

	cc = []string{goEnv("CC")}

	out := goEnv("GOGCCFLAGS")
	quote := '\000'
	start := 0
	lastSpace := true
	backslash := false
	s := string(out)
	for i, c := range s {
		if quote == '\000' && unicode.IsSpace(c) {
			if !lastSpace {
				cc = append(cc, s[start:i])
				lastSpace = true
			}
		} else {
			if lastSpace {
				start = i
				lastSpace = false
			}
			if quote == '\000' && !backslash && (c == '"' || c == '\'') {
				quote = c
				backslash = false
			} else if !backslash && quote == c {
				quote = '\000'
			} else if (quote == '\000' || quote == '"') && !backslash && c == '\\' {
				backslash = true
			} else {
				backslash = false
			}
		}
	}
	if !lastSpace {
		cc = append(cc, s[start:])
	}

	switch GOOS {
	case "darwin", "ios":
		// For Darwin/ARM.
		// TODO(crawshaw): can we do better?
		cc = append(cc, []string{"-framework", "CoreFoundation", "-framework", "Foundation"}...)
	case "android":
		cc = append(cc, "-pie")
	}
	libgodir := GOOS + "_" + GOARCH
	switch GOOS {
	case "darwin", "ios":
		if GOARCH == "arm64" {
			libgodir += "_shared"
		}
	case "dragonfly", "freebsd", "linux", "netbsd", "openbsd", "solaris", "illumos":
		libgodir += "_shared"
	}
	cc = append(cc, "-I", filepath.Join("pkg", libgodir))

	if GOOS == "windows" {
		exeSuffix = ".exe"
	}

	// Copy testdata into GOPATH/src/testcshared, along with a go.mod file
	// declaring the same path.

	GOPATH, err := ioutil.TempDir("", "cshared_test")
	if err != nil {
		log.Panic(err)
	}
	defer os.RemoveAll(GOPATH)
	os.Setenv("GOPATH", GOPATH)

	modRoot := filepath.Join(GOPATH, "src", "testcshared")
	if err := overlayDir(modRoot, "testdata"); err != nil {
		log.Panic(err)
	}
	if err := os.Chdir(modRoot); err != nil {
		log.Panic(err)
	}
	os.Setenv("PWD", modRoot)
	if err := ioutil.WriteFile("go.mod", []byte("module testcshared\n"), 0666); err != nil {
		log.Panic(err)
	}

	// Directory where cgo headers and outputs will be installed.
	// The installation directory format varies depending on the platform.
	output, err := exec.Command("go", "list",
		"-buildmode=c-shared",
		"-installsuffix", "testcshared",
		"-f", "{{.Target}}",
		"./libgo").CombinedOutput()
	if err != nil {
		log.Panicf("go list failed: %v\n%s", err, output)
	}
	target := string(bytes.TrimSpace(output))
	libgoname = filepath.Base(target)
	installdir = filepath.Dir(target)
	libSuffix = strings.TrimPrefix(filepath.Ext(target), ".")

	return m.Run()
}

func goEnv(key string) string {
	out, err := exec.Command("go", "env", key).Output()
	if err != nil {
		log.Printf("go env %s failed:\n%s", key, err)
		log.Panicf("%s", err.(*exec.ExitError).Stderr)
	}
	return strings.TrimSpace(string(out))
}

func cmdToRun(name string) string {
	return "./" + name + exeSuffix
}

func adbCmd() []string {
	cmd := []string{"adb"}
	if flags := os.Getenv("GOANDROID_ADB_FLAGS"); flags != "" {
		cmd = append(cmd, strings.Split(flags, " ")...)
	}
	return cmd
}

func adbPush(t *testing.T, filename string) {
	if runtime.GOOS == GOOS || GOOS != "android" {
		return
	}
	args := append(adbCmd(), "push", filename, fmt.Sprintf("%s/%s", androiddir, filename))
	cmd := exec.Command(args[0], args[1:]...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("adb command failed: %v\n%s\n", err, out)
	}
}

func adbRun(t *testing.T, env []string, adbargs ...string) string {
	if GOOS != "android" {
		t.Fatalf("trying to run adb command when operating system is not android.")
	}
	args := append(adbCmd(), "exec-out")
	// Propagate LD_LIBRARY_PATH to the adb shell invocation.
	for _, e := range env {
		if strings.Index(e, "LD_LIBRARY_PATH=") != -1 {
			adbargs = append([]string{e}, adbargs...)
			break
		}
	}
	shellcmd := fmt.Sprintf("cd %s; %s", androiddir, strings.Join(adbargs, " "))
	args = append(args, shellcmd)
	cmd := exec.Command(args[0], args[1:]...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("adb command failed: %v\n%s\n", err, out)
	}
	return strings.Replace(string(out), "\r", "", -1)
}

func run(t *testing.T, extraEnv []string, args ...string) string {
	t.Helper()
	cmd := exec.Command(args[0], args[1:]...)
	if len(extraEnv) > 0 {
		cmd.Env = append(os.Environ(), extraEnv...)
	}

	if GOOS != "windows" {
		// TestUnexportedSymbols relies on file descriptor 30
		// being closed when the program starts, so enforce
		// that in all cases. (The first three descriptors are
		// stdin/stdout/stderr, so we just need to make sure
		// that cmd.ExtraFiles[27] exists and is nil.)
		cmd.ExtraFiles = make([]*os.File, 28)
	}

	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("command failed: %v\n%v\n%s\n", args, err, out)
	} else {
		t.Logf("run: %v", args)
	}
	return string(out)
}

func runExe(t *testing.T, extraEnv []string, args ...string) string {
	t.Helper()
	if runtime.GOOS != GOOS && GOOS == "android" {
		return adbRun(t, append(os.Environ(), extraEnv...), args...)
	}
	return run(t, extraEnv, args...)
}

func runCC(t *testing.T, args ...string) string {
	t.Helper()
	// This function is run in parallel, so append to a copy of cc
	// rather than cc itself.
	return run(t, nil, append(append([]string(nil), cc...), args...)...)
}

func createHeaders() error {
	// The 'cgo' command generates a number of additional artifacts,
	// but we're only interested in the header.
	// Shunt the rest of the outputs to a temporary directory.
	objDir, err := ioutil.TempDir("", "testcshared_obj")
	if err != nil {
		return err
	}
	defer os.RemoveAll(objDir)

	// Generate a C header file for p, which is a non-main dependency
	// of main package libgo.
	//
	// TODO(golang.org/issue/35715): This should be simpler.
	args := []string{"go", "tool", "cgo",
		"-objdir", objDir,
		"-exportheader", "p.h",
		filepath.Join(".", "p", "p.go")}
	cmd := exec.Command(args[0], args[1:]...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("command failed: %v\n%v\n%s\n", args, err, out)
	}

	// Generate a C header file for libgo itself.
	args = []string{"go", "install", "-buildmode=c-shared",
		"-installsuffix", "testcshared", "./libgo"}
	cmd = exec.Command(args[0], args[1:]...)
	out, err = cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("command failed: %v\n%v\n%s\n", args, err, out)
	}

	args = []string{"go", "build", "-buildmode=c-shared",
		"-installsuffix", "testcshared",
		"-o", libgoname,
		filepath.Join(".", "libgo", "libgo.go")}
	cmd = exec.Command(args[0], args[1:]...)
	out, err = cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("command failed: %v\n%v\n%s\n", args, err, out)
	}

	if runtime.GOOS != GOOS && GOOS == "android" {
		args = append(adbCmd(), "push", libgoname, fmt.Sprintf("%s/%s", androiddir, libgoname))
		cmd = exec.Command(args[0], args[1:]...)
		out, err = cmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("adb command failed: %v\n%s\n", err, out)
		}
	}

	return nil
}

var (
	headersOnce sync.Once
	headersErr  error
)

func createHeadersOnce(t *testing.T) {
	headersOnce.Do(func() {
		headersErr = createHeaders()
	})
	if headersErr != nil {
		t.Fatal(headersErr)
	}
}

func cleanupAndroid() {
	if GOOS != "android" {
		return
	}
	args := append(adbCmd(), "exec-out", "rm", "-rf", androiddir)
	cmd := exec.Command(args[0], args[1:]...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Panicf("cleanupAndroid failed: %v\n%s\n", err, out)
	}
}

// test0: exported symbols in shared lib are accessible.
func TestExportedSymbols(t *testing.T) {
	t.Parallel()

	cmd := "testp0"
	bin := cmdToRun(cmd)

	createHeadersOnce(t)

	runCC(t, "-I", installdir, "-o", cmd, "main0.c", libgoname)
	adbPush(t, cmd)

	defer os.Remove(bin)

	out := runExe(t, []string{"LD_LIBRARY_PATH=."}, bin)
	if strings.TrimSpace(out) != "PASS" {
		t.Error(out)
	}
}

func checkNumberOfExportedFunctionsWindows(t *testing.T, exportAllSymbols bool) {
	const prog = `
package main

import "C"

//export GoFunc
func GoFunc() {
	println(42)
}

//export GoFunc2
func GoFunc2() {
	println(24)
}

func main() {
}
`

	tmpdir := t.TempDir()

	srcfile := filepath.Join(tmpdir, "test.go")
	objfile := filepath.Join(tmpdir, "test.dll")
	if err := ioutil.WriteFile(srcfile, []byte(prog), 0666); err != nil {
		t.Fatal(err)
	}
	argv := []string{"build", "-buildmode=c-shared"}
	if exportAllSymbols {
		argv = append(argv, "-ldflags", "-extldflags=-Wl,--export-all-symbols")
	}
	argv = append(argv, "-o", objfile, srcfile)
	out, err := exec.Command("go", argv...).CombinedOutput()
	if err != nil {
		t.Fatalf("build failure: %s\n%s\n", err, string(out))
	}

	f, err := pe.Open(objfile)
	if err != nil {
		t.Fatalf("pe.Open failed: %v", err)
	}
	defer f.Close()
	section := f.Section(".edata")
	if section == nil {
		t.Fatalf(".edata section is not present")
	}

	// TODO: deduplicate this struct from cmd/link/internal/ld/pe.go
	type IMAGE_EXPORT_DIRECTORY struct {
		_                 [2]uint32
		_                 [2]uint16
		_                 [2]uint32
		NumberOfFunctions uint32
		NumberOfNames     uint32
		_                 [3]uint32
	}
	var e IMAGE_EXPORT_DIRECTORY
	if err := binary.Read(section.Open(), binary.LittleEndian, &e); err != nil {
		t.Fatalf("binary.Read failed: %v", err)
	}

	// Only the two exported functions and _cgo_dummy_export should be exported
	expectedNumber := uint32(3)

	if exportAllSymbols {
		if e.NumberOfFunctions <= expectedNumber {
			t.Fatalf("missing exported functions: %v", e.NumberOfFunctions)
		}
		if e.NumberOfNames <= expectedNumber {
			t.Fatalf("missing exported names: %v", e.NumberOfNames)
		}
	} else {
		if e.NumberOfFunctions != expectedNumber {
			t.Fatalf("got %d exported functions; want %d", e.NumberOfFunctions, expectedNumber)
		}
		if e.NumberOfNames != expectedNumber {
			t.Fatalf("got %d exported names; want %d", e.NumberOfNames, expectedNumber)
		}
	}
}

func TestNumberOfExportedFunctions(t *testing.T) {
	if GOOS != "windows" {
		t.Skip("skipping windows only test")
	}
	t.Parallel()

	t.Run("OnlyExported", func(t *testing.T) {
		checkNumberOfExportedFunctionsWindows(t, false)
	})
	t.Run("All", func(t *testing.T) {
		checkNumberOfExportedFunctionsWindows(t, true)
	})
}

// test1: shared library can be dynamically loaded and exported symbols are accessible.
func TestExportedSymbolsWithDynamicLoad(t *testing.T) {
	t.Parallel()

	if GOOS == "windows" {
		t.Logf("Skipping on %s", GOOS)
		return
	}

	cmd := "testp1"
	bin := cmdToRun(cmd)

	createHeadersOnce(t)

	if GOOS != "freebsd" {
		runCC(t, "-o", cmd, "main1.c", "-ldl")
	} else {
		runCC(t, "-o", cmd, "main1.c")
	}
	adbPush(t, cmd)

	defer os.Remove(bin)

	out := runExe(t, nil, bin, "./"+libgoname)
	if strings.TrimSpace(out) != "PASS" {
		t.Error(out)
	}
}

// test2: tests libgo2 which does not export any functions.
func TestUnexportedSymbols(t *testing.T) {
	t.Parallel()

	if GOOS == "windows" {
		t.Logf("Skipping on %s", GOOS)
		return
	}

	cmd := "testp2"
	bin := cmdToRun(cmd)
	libname := "libgo2." + libSuffix

	run(t,
		nil,
		"go", "build",
		"-buildmode=c-shared",
		"-installsuffix", "testcshared",
		"-o", libname, "./libgo2",
	)
	adbPush(t, libname)

	linkFlags := "-Wl,--no-as-needed"
	if GOOS == "darwin" || GOOS == "ios" {
		linkFlags = ""
	}

	runCC(t, "-o", cmd, "main2.c", linkFlags, libname)
	adbPush(t, cmd)

	defer os.Remove(libname)
	defer os.Remove(bin)

	out := runExe(t, []string{"LD_LIBRARY_PATH=."}, bin)

	if strings.TrimSpace(out) != "PASS" {
		t.Error(out)
	}
}

// test3: tests main.main is exported on android.
func TestMainExportedOnAndroid(t *testing.T) {
	t.Parallel()

	switch GOOS {
	case "android":
		break
	default:
		t.Logf("Skipping on %s", GOOS)
		return
	}

	cmd := "testp3"
	bin := cmdToRun(cmd)

	createHeadersOnce(t)

	runCC(t, "-o", cmd, "main3.c", "-ldl")
	adbPush(t, cmd)

	defer os.Remove(bin)

	out := runExe(t, nil, bin, "./"+libgoname)
	if strings.TrimSpace(out) != "PASS" {
		t.Error(out)
	}
}

func testSignalHandlers(t *testing.T, pkgname, cfile, cmd string) {
	libname := pkgname + "." + libSuffix
	run(t,
		nil,
		"go", "build",
		"-buildmode=c-shared",
		"-installsuffix", "testcshared",
		"-o", libname, pkgname,
	)
	adbPush(t, libname)
	if GOOS != "freebsd" {
		runCC(t, "-pthread", "-o", cmd, cfile, "-ldl")
	} else {
		runCC(t, "-pthread", "-o", cmd, cfile)
	}
	adbPush(t, cmd)

	bin := cmdToRun(cmd)

	defer os.Remove(libname)
	defer os.Remove(bin)
	defer os.Remove(pkgname + ".h")

	out := runExe(t, nil, bin, "./"+libname)
	if strings.TrimSpace(out) != "PASS" {
		t.Error(run(t, nil, bin, libname, "verbose"))
	}
}

// test4: test signal handlers
func TestSignalHandlers(t *testing.T) {
	t.Parallel()
	if GOOS == "windows" {
		t.Logf("Skipping on %s", GOOS)
		return
	}
	testSignalHandlers(t, "./libgo4", "main4.c", "testp4")
}

// test5: test signal handlers with os/signal.Notify
func TestSignalHandlersWithNotify(t *testing.T) {
	t.Parallel()
	if GOOS == "windows" {
		t.Logf("Skipping on %s", GOOS)
		return
	}
	testSignalHandlers(t, "./libgo5", "main5.c", "testp5")
}

func TestPIE(t *testing.T) {
	t.Parallel()

	switch GOOS {
	case "linux", "android":
		break
	default:
		t.Logf("Skipping on %s", GOOS)
		return
	}

	createHeadersOnce(t)

	f, err := elf.Open(libgoname)
	if err != nil {
		t.Fatalf("elf.Open failed: %v", err)
	}
	defer f.Close()

	ds := f.SectionByType(elf.SHT_DYNAMIC)
	if ds == nil {
		t.Fatalf("no SHT_DYNAMIC section")
	}
	d, err := ds.Data()
	if err != nil {
		t.Fatalf("can't read SHT_DYNAMIC contents: %v", err)
	}
	for len(d) > 0 {
		var tag elf.DynTag
		switch f.Class {
		case elf.ELFCLASS32:
			tag = elf.DynTag(f.ByteOrder.Uint32(d[:4]))
			d = d[8:]
		case elf.ELFCLASS64:
			tag = elf.DynTag(f.ByteOrder.Uint64(d[:8]))
			d = d[16:]
		}
		if tag == elf.DT_TEXTREL {
			t.Fatalf("%s has DT_TEXTREL flag", libgoname)
		}
	}
}

// Test that installing a second time recreates the header file.
func TestCachedInstall(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "cshared")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	copyFile(t, filepath.Join(tmpdir, "src", "testcshared", "go.mod"), "go.mod")
	copyFile(t, filepath.Join(tmpdir, "src", "testcshared", "libgo", "libgo.go"), filepath.Join("libgo", "libgo.go"))
	copyFile(t, filepath.Join(tmpdir, "src", "testcshared", "p", "p.go"), filepath.Join("p", "p.go"))

	env := append(os.Environ(), "GOPATH="+tmpdir, "GOBIN="+filepath.Join(tmpdir, "bin"))

	buildcmd := []string{"go", "install", "-x", "-buildmode=c-shared", "-installsuffix", "testcshared", "./libgo"}

	cmd := exec.Command(buildcmd[0], buildcmd[1:]...)
	cmd.Dir = filepath.Join(tmpdir, "src", "testcshared")
	cmd.Env = env
	t.Log(buildcmd)
	out, err := cmd.CombinedOutput()
	t.Logf("%s", out)
	if err != nil {
		t.Fatal(err)
	}

	var libgoh, ph string

	walker := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			t.Fatal(err)
		}
		var ps *string
		switch filepath.Base(path) {
		case "libgo.h":
			ps = &libgoh
		case "p.h":
			ps = &ph
		}
		if ps != nil {
			if *ps != "" {
				t.Fatalf("%s found again", *ps)
			}
			*ps = path
		}
		return nil
	}

	if err := filepath.Walk(tmpdir, walker); err != nil {
		t.Fatal(err)
	}

	if libgoh == "" {
		t.Fatal("libgo.h not installed")
	}

	if err := os.Remove(libgoh); err != nil {
		t.Fatal(err)
	}

	cmd = exec.Command(buildcmd[0], buildcmd[1:]...)
	cmd.Dir = filepath.Join(tmpdir, "src", "testcshared")
	cmd.Env = env
	t.Log(buildcmd)
	out, err = cmd.CombinedOutput()
	t.Logf("%s", out)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := os.Stat(libgoh); err != nil {
		t.Errorf("libgo.h not installed in second run: %v", err)
	}
}

// copyFile copies src to dst.
func copyFile(t *testing.T, dst, src string) {
	t.Helper()
	data, err := ioutil.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0777); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(dst, data, 0666); err != nil {
		t.Fatal(err)
	}
}

func TestGo2C2Go(t *testing.T) {
	switch GOOS {
	case "darwin", "ios":
		// Darwin shared libraries don't support the multiple
		// copies of the runtime package implied by this test.
		t.Skip("linking c-shared into Go programs not supported on Darwin; issue 29061")
	case "android":
		t.Skip("test fails on android; issue 29087")
	}

	t.Parallel()

	tmpdir, err := ioutil.TempDir("", "cshared-TestGo2C2Go")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	lib := filepath.Join(tmpdir, "libtestgo2c2go."+libSuffix)
	run(t, nil, "go", "build", "-buildmode=c-shared", "-o", lib, "./go2c2go/go")

	cgoCflags := os.Getenv("CGO_CFLAGS")
	if cgoCflags != "" {
		cgoCflags += " "
	}
	cgoCflags += "-I" + tmpdir

	cgoLdflags := os.Getenv("CGO_LDFLAGS")
	if cgoLdflags != "" {
		cgoLdflags += " "
	}
	cgoLdflags += "-L" + tmpdir + " -ltestgo2c2go"

	goenv := []string{"CGO_CFLAGS=" + cgoCflags, "CGO_LDFLAGS=" + cgoLdflags}

	ldLibPath := os.Getenv("LD_LIBRARY_PATH")
	if ldLibPath != "" {
		ldLibPath += ":"
	}
	ldLibPath += tmpdir

	runenv := []string{"LD_LIBRARY_PATH=" + ldLibPath}

	bin := filepath.Join(tmpdir, "m1") + exeSuffix
	run(t, goenv, "go", "build", "-o", bin, "./go2c2go/m1")
	runExe(t, runenv, bin)

	bin = filepath.Join(tmpdir, "m2") + exeSuffix
	run(t, goenv, "go", "build", "-o", bin, "./go2c2go/m2")
	runExe(t, runenv, bin)
}
