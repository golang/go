// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cshared_test

import (
	"bufio"
	"bytes"
	"cmd/cgo/internal/cgotest"
	"cmp"
	"debug/elf"
	"debug/pe"
	"encoding/binary"
	"flag"
	"fmt"
	"internal/testenv"
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

var globalSkip = func(t *testing.T) {}

// C compiler with args (from $(go env CC) $(go env GOGCCFLAGS)).
var cc []string

// ".exe" on Windows.
var exeSuffix string

var GOOS, GOARCH, GOROOT string
var installdir string
var libgoname string

func TestMain(m *testing.M) {
	os.Exit(testMain(m))
}

func testMain(m *testing.M) int {
	log.SetFlags(log.Lshortfile)
	flag.Parse()
	if testing.Short() && testenv.Builder() == "" {
		globalSkip = func(t *testing.T) { t.Skip("short mode and $GO_BUILDER_NAME not set") }
		return m.Run()
	}
	if runtime.GOOS == "linux" {
		if _, err := os.Stat("/etc/alpine-release"); err == nil {
			globalSkip = func(t *testing.T) { t.Skip("skipping failing test on alpine - go.dev/issue/19938") }
			return m.Run()
		}
	}
	if !testenv.HasGoBuild() {
		// Checking for "go build" is a proxy for whether or not we can run "go env".
		globalSkip = func(t *testing.T) { t.Skip("no go build") }
		return m.Run()
	}

	GOOS = goEnv("GOOS")
	GOARCH = goEnv("GOARCH")
	GOROOT = goEnv("GOROOT")

	if _, err := os.Stat(GOROOT); os.IsNotExist(err) {
		log.Fatalf("Unable able to find GOROOT at '%s'", GOROOT)
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

	// Force reallocation (and avoid aliasing bugs) for parallel tests that append to cc.
	cc = cc[:len(cc):len(cc)]

	if GOOS == "windows" {
		exeSuffix = ".exe"
	}

	// Copy testdata into GOPATH/src/testcshared, along with a go.mod file
	// declaring the same path.

	GOPATH, err := os.MkdirTemp("", "cshared_test")
	if err != nil {
		log.Panic(err)
	}
	defer os.RemoveAll(GOPATH)
	os.Setenv("GOPATH", GOPATH)

	modRoot := filepath.Join(GOPATH, "src", "testcshared")
	if err := cgotest.OverlayDir(modRoot, "testdata"); err != nil {
		log.Panic(err)
	}
	if err := os.Chdir(modRoot); err != nil {
		log.Panic(err)
	}
	os.Setenv("PWD", modRoot)
	if err := os.WriteFile("go.mod", []byte("module testcshared\n"), 0666); err != nil {
		log.Panic(err)
	}

	defer func() {
		if installdir != "" {
			err := os.RemoveAll(installdir)
			if err != nil {
				log.Panic(err)
			}
		}
	}()

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

func run(t *testing.T, extraEnv []string, args ...string) string {
	t.Helper()
	cmd := exec.Command(args[0], args[1:]...)
	if len(extraEnv) > 0 {
		cmd.Env = append(os.Environ(), extraEnv...)
	}
	stderr := new(strings.Builder)
	cmd.Stderr = stderr

	if GOOS != "windows" {
		// TestUnexportedSymbols relies on file descriptor 30
		// being closed when the program starts, so enforce
		// that in all cases. (The first three descriptors are
		// stdin/stdout/stderr, so we just need to make sure
		// that cmd.ExtraFiles[27] exists and is nil.)
		cmd.ExtraFiles = make([]*os.File, 28)
	}

	t.Logf("run: %v", args)
	out, err := cmd.Output()
	if stderr.Len() > 0 {
		t.Logf("stderr:\n%s", stderr)
	}
	if err != nil {
		t.Fatalf("command failed: %v\n%v\n%s\n", args, err, out)
	}
	return string(out)
}

func runExe(t *testing.T, extraEnv []string, args ...string) string {
	t.Helper()
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
	objDir, err := os.MkdirTemp("", "testcshared_obj")
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
	installdir, err = os.MkdirTemp("", "testcshared")
	if err != nil {
		return err
	}
	libgoname = "libgo.a"

	args = []string{"go", "build", "-buildmode=c-shared", "-o", filepath.Join(installdir, libgoname), "./libgo"}
	cmd = exec.Command(args[0], args[1:]...)
	out, err = cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("command failed: %v\n%v\n%s\n", args, err, out)
	}

	args = []string{"go", "build", "-buildmode=c-shared",
		"-installsuffix", "testcshared",
		"-o", libgoname,
		filepath.Join(".", "libgo", "libgo.go")}
	if GOOS == "windows" && strings.HasSuffix(args[6], ".a") {
		args[6] = strings.TrimSuffix(args[6], ".a") + ".dll"
	}
	cmd = exec.Command(args[0], args[1:]...)
	out, err = cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("command failed: %v\n%v\n%s\n", args, err, out)
	}
	if GOOS == "windows" {
		// We can't simply pass -Wl,--out-implib, because this relies on having imports from multiple packages,
		// which results in the linkers output implib getting overwritten at each step. So instead build the
		// import library the traditional way, using a def file.
		err = os.WriteFile("libgo.def",
			[]byte("LIBRARY libgo.dll\nEXPORTS\n\tDidInitRun\n\tDidMainRun\n\tDivu\n\tFromPkg\n"),
			0644)
		if err != nil {
			return fmt.Errorf("unable to write def file: %v", err)
		}
		out, err = exec.Command(cc[0], append(cc[1:], "-print-prog-name=dlltool")...).CombinedOutput()
		if err != nil {
			return fmt.Errorf("unable to find dlltool path: %v\n%s\n", err, out)
		}
		dlltoolpath := strings.TrimSpace(string(out))
		if filepath.Ext(dlltoolpath) == "" {
			// Some compilers report slash-separated paths without extensions
			// instead of ordinary Windows paths.
			// Try to find the canonical name for the path.
			if lp, err := exec.LookPath(dlltoolpath); err == nil {
				dlltoolpath = lp
			}
		}

		args := []string{dlltoolpath, "-D", args[6], "-l", libgoname, "-d", "libgo.def"}

		if filepath.Ext(dlltoolpath) == "" {
			// This is an unfortunate workaround for
			// https://github.com/mstorsjo/llvm-mingw/issues/205 in which
			// we basically reimplement the contents of the dlltool.sh
			// wrapper: https://git.io/JZFlU.
			// TODO(thanm): remove this workaround once we can upgrade
			// the compilers on the windows-arm64 builder.
			dlltoolContents, err := os.ReadFile(args[0])
			if err != nil {
				return fmt.Errorf("unable to read dlltool: %v\n", err)
			}
			if bytes.HasPrefix(dlltoolContents, []byte("#!/bin/sh")) && bytes.Contains(dlltoolContents, []byte("llvm-dlltool")) {
				base, name := filepath.Split(args[0])
				args[0] = filepath.Join(base, "llvm-dlltool")
				var machine string
				switch prefix, _, _ := strings.Cut(name, "-"); prefix {
				case "i686":
					machine = "i386"
				case "x86_64":
					machine = "i386:x86-64"
				case "armv7":
					machine = "arm"
				case "aarch64":
					machine = "arm64"
				}
				if len(machine) > 0 {
					args = append(args, "-m", machine)
				}
			}
		}

		out, err = exec.Command(args[0], args[1:]...).CombinedOutput()
		if err != nil {
			return fmt.Errorf("unable to run dlltool to create import library: %v\n%s\n", err, out)
		}
	}

	return nil
}

var (
	headersOnce sync.Once
	headersErr  error
)

func createHeadersOnce(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveBuildMode(t, "c-shared")

	headersOnce.Do(func() {
		headersErr = createHeaders()
	})
	if headersErr != nil {
		t.Helper()
		t.Fatal(headersErr)
	}
}

// test0: exported symbols in shared lib are accessible.
func TestExportedSymbols(t *testing.T) {
	globalSkip(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveExec(t)

	t.Parallel()

	cmd := "testp0"
	bin := cmdToRun(cmd)

	createHeadersOnce(t)

	runCC(t, "-I", installdir, "-o", cmd, "main0.c", libgoname)

	defer os.Remove(bin)

	out := runExe(t, []string{"LD_LIBRARY_PATH=."}, bin)
	if strings.TrimSpace(out) != "PASS" {
		t.Error(out)
	}
}

func checkNumberOfExportedSymbolsWindows(t *testing.T, exportedSymbols int, wantAll bool) {
	t.Parallel()
	tmpdir := t.TempDir()

	prog := `
package main
import "C"
func main() {}
`

	for i := range exportedSymbols {
		prog += fmt.Sprintf(`
//export GoFunc%d
func GoFunc%d() {}
`, i, i)
	}

	srcfile := filepath.Join(tmpdir, "test.go")
	objfile := filepath.Join(tmpdir, "test.dll")
	if err := os.WriteFile(srcfile, []byte(prog), 0666); err != nil {
		t.Fatal(err)
	}
	argv := []string{"build", "-buildmode=c-shared"}
	if wantAll {
		argv = append(argv, "-ldflags", "-extldflags=-Wl,--export-all-symbols")
	}
	argv = append(argv, "-o", objfile, srcfile)
	out, err := exec.Command(testenv.GoToolPath(t), argv...).CombinedOutput()
	if err != nil {
		t.Fatalf("build failure: %s\n%s\n", err, string(out))
	}

	f, err := pe.Open(objfile)
	if err != nil {
		t.Fatalf("pe.Open failed: %v", err)
	}
	defer f.Close()

	_, pe64 := f.OptionalHeader.(*pe.OptionalHeader64)
	// grab the export data directory entry
	var idd pe.DataDirectory
	if pe64 {
		idd = f.OptionalHeader.(*pe.OptionalHeader64).DataDirectory[pe.IMAGE_DIRECTORY_ENTRY_EXPORT]
	} else {
		idd = f.OptionalHeader.(*pe.OptionalHeader32).DataDirectory[pe.IMAGE_DIRECTORY_ENTRY_EXPORT]
	}

	// figure out which section contains the import directory table
	var section *pe.Section
	for _, s := range f.Sections {
		if s.Offset == 0 {
			continue
		}
		if s.VirtualAddress <= idd.VirtualAddress && idd.VirtualAddress-s.VirtualAddress < s.VirtualSize {
			section = s
			break
		}
	}
	if section == nil {
		t.Fatal("no section contains export directory")
	}
	d, err := section.Data()
	if err != nil {
		t.Fatal(err)
	}
	// seek to the virtual address specified in the export data directory
	d = d[idd.VirtualAddress-section.VirtualAddress:]

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
	if err := binary.Read(bytes.NewReader(d), binary.LittleEndian, &e); err != nil {
		t.Fatalf("binary.Read failed: %v", err)
	}

	exportedSymbols = cmp.Or(exportedSymbols, 1) // _cgo_stub_export is exported if there are no other symbols exported

	// NumberOfNames is the number of functions exported with a unique name.
	// NumberOfFunctions can be higher than that because it also counts
	// functions exported only by ordinal, a unique number asigned by the linker,
	// and linkers might add an unknown number of their own ordinal-only functions.
	if wantAll {
		if e.NumberOfNames <= uint32(exportedSymbols) {
			t.Errorf("got %d exported names, want > %d", e.NumberOfNames, exportedSymbols)
		}
	} else {
		if e.NumberOfNames != uint32(exportedSymbols) {
			t.Errorf("got %d exported names, want %d", e.NumberOfNames, exportedSymbols)
		}
	}
}

func TestNumberOfExportedFunctions(t *testing.T) {
	if GOOS != "windows" {
		t.Skip("skipping windows only test")
	}
	globalSkip(t)
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveBuildMode(t, "c-shared")

	t.Parallel()

	for i := range 3 {
		t.Run(fmt.Sprintf("OnlyExported/%d", i), func(t *testing.T) {
			checkNumberOfExportedSymbolsWindows(t, i, false)
		})
		t.Run(fmt.Sprintf("All/%d", i), func(t *testing.T) {
			checkNumberOfExportedSymbolsWindows(t, i, true)
		})
	}
}

// test1: shared library can be dynamically loaded and exported symbols are accessible.
func TestExportedSymbolsWithDynamicLoad(t *testing.T) {
	if GOOS == "windows" {
		t.Skipf("Skipping on %s", GOOS)
	}
	globalSkip(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveExec(t)

	t.Parallel()

	cmd := "testp1"
	bin := cmdToRun(cmd)

	createHeadersOnce(t)

	if GOOS != "freebsd" {
		runCC(t, "-o", cmd, "main1.c", "-ldl")
	} else {
		runCC(t, "-o", cmd, "main1.c")
	}

	defer os.Remove(bin)

	out := runExe(t, nil, bin, "./"+libgoname)
	if strings.TrimSpace(out) != "PASS" {
		t.Error(out)
	}
}

// test2: tests libgo2 which does not export any functions.
func TestUnexportedSymbols(t *testing.T) {
	if GOOS == "windows" {
		t.Skipf("Skipping on %s", GOOS)
	}
	globalSkip(t)
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveBuildMode(t, "c-shared")

	t.Parallel()

	cmd := "testp2"
	bin := cmdToRun(cmd)
	libname := "libgo2.a"

	run(t,
		nil,
		"go", "build",
		"-buildmode=c-shared",
		"-installsuffix", "testcshared",
		"-o", libname, "./libgo2",
	)

	linkFlags := "-Wl,--no-as-needed"
	if GOOS == "darwin" || GOOS == "ios" {
		linkFlags = ""
	}

	runCC(t, "-o", cmd, "main2.c", linkFlags, libname)

	defer os.Remove(libname)
	defer os.Remove(bin)

	out := runExe(t, []string{"LD_LIBRARY_PATH=."}, bin)

	if strings.TrimSpace(out) != "PASS" {
		t.Error(out)
	}
}

// test3: tests main.main is exported on android.
func TestMainExportedOnAndroid(t *testing.T) {
	globalSkip(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveExec(t)

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

	defer os.Remove(bin)

	out := runExe(t, nil, bin, "./"+libgoname)
	if strings.TrimSpace(out) != "PASS" {
		t.Error(out)
	}
}

func testSignalHandlers(t *testing.T, pkgname, cfile, cmd string) {
	if GOOS == "windows" {
		t.Skipf("Skipping on %s", GOOS)
	}
	globalSkip(t)
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveBuildMode(t, "c-shared")

	libname := pkgname + ".a"
	run(t,
		nil,
		"go", "build",
		"-buildmode=c-shared",
		"-installsuffix", "testcshared",
		"-o", libname, pkgname,
	)
	if GOOS != "freebsd" {
		runCC(t, "-pthread", "-o", cmd, cfile, "-ldl")
	} else {
		runCC(t, "-pthread", "-o", cmd, cfile)
	}

	bin := cmdToRun(cmd)

	defer os.Remove(libname)
	defer os.Remove(bin)
	defer os.Remove(pkgname + ".h")

	args := []string{bin, "./" + libname}
	if testing.Verbose() {
		args = append(args, "verbose")
	}
	out := runExe(t, nil, args...)
	if strings.TrimSpace(out) != "PASS" {
		t.Errorf("%v%s", args, out)
	}
}

// test4: test signal handlers
func TestSignalHandlers(t *testing.T) {
	t.Parallel()
	testSignalHandlers(t, "./libgo4", "main4.c", "testp4")
}

// test5: test signal handlers with os/signal.Notify
func TestSignalHandlersWithNotify(t *testing.T) {
	t.Parallel()
	testSignalHandlers(t, "./libgo5", "main5.c", "testp5")
}

func TestPIE(t *testing.T) {
	switch GOOS {
	case "linux", "android":
		break
	default:
		t.Skipf("Skipping on %s", GOOS)
	}
	globalSkip(t)

	t.Parallel()

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
	globalSkip(t)
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveBuildMode(t, "c-shared")

	tmpdir, err := os.MkdirTemp("", "cshared")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	copyFile(t, filepath.Join(tmpdir, "src", "testcshared", "go.mod"), "go.mod")
	copyFile(t, filepath.Join(tmpdir, "src", "testcshared", "libgo", "libgo.go"), filepath.Join("libgo", "libgo.go"))
	copyFile(t, filepath.Join(tmpdir, "src", "testcshared", "p", "p.go"), filepath.Join("p", "p.go"))

	buildcmd := []string{"go", "install", "-x", "-buildmode=c-shared", "-installsuffix", "testcshared", "./libgo"}

	cmd := exec.Command(buildcmd[0], buildcmd[1:]...)
	cmd.Dir = filepath.Join(tmpdir, "src", "testcshared")
	env := append(cmd.Environ(),
		"GOPATH="+tmpdir,
		"GOBIN="+filepath.Join(tmpdir, "bin"),
		"GO111MODULE=off", // 'go install' only works in GOPATH mode
	)
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
	data, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dst, data, 0666); err != nil {
		t.Fatal(err)
	}
}

func TestGo2C2Go(t *testing.T) {
	switch GOOS {
	case "darwin", "ios", "windows":
		// Non-ELF shared libraries don't support the multiple
		// copies of the runtime package implied by this test.
		t.Skipf("linking c-shared into Go programs not supported on %s; issue 29061, 49457", GOOS)
	case "android":
		t.Skip("test fails on android; issue 29087")
	}
	globalSkip(t)
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveBuildMode(t, "c-shared")

	t.Parallel()

	tmpdir, err := os.MkdirTemp("", "cshared-TestGo2C2Go")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	lib := filepath.Join(tmpdir, "libtestgo2c2go.a")
	var env []string
	if GOOS == "windows" && strings.HasSuffix(lib, ".a") {
		env = append(env, "CGO_LDFLAGS=-Wl,--out-implib,"+lib, "CGO_LDFLAGS_ALLOW=.*")
		lib = strings.TrimSuffix(lib, ".a") + ".dll"
	}
	run(t, env, "go", "build", "-buildmode=c-shared", "-o", lib, "./go2c2go/go")

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

func TestIssue36233(t *testing.T) {
	globalSkip(t)
	testenv.MustHaveCGO(t)

	t.Parallel()

	// Test that the export header uses GoComplex64 and GoComplex128
	// for complex types.

	tmpdir, err := os.MkdirTemp("", "cshared-TestIssue36233")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	const exportHeader = "issue36233.h"

	run(t, nil, "go", "tool", "cgo", "-exportheader", exportHeader, "-objdir", tmpdir, "./issue36233/issue36233.go")
	data, err := os.ReadFile(exportHeader)
	if err != nil {
		t.Fatal(err)
	}

	funcs := []struct{ name, signature string }{
		{"exportComplex64", "GoComplex64 exportComplex64(GoComplex64 v)"},
		{"exportComplex128", "GoComplex128 exportComplex128(GoComplex128 v)"},
		{"exportComplexfloat", "GoComplex64 exportComplexfloat(GoComplex64 v)"},
		{"exportComplexdouble", "GoComplex128 exportComplexdouble(GoComplex128 v)"},
	}

	scanner := bufio.NewScanner(bytes.NewReader(data))
	var found int
	for scanner.Scan() {
		b := scanner.Bytes()
		for _, fn := range funcs {
			if bytes.Contains(b, []byte(fn.name)) {
				found++
				if !bytes.Contains(b, []byte(fn.signature)) {
					t.Errorf("function signature mismatch; got %q, want %q", b, fn.signature)
				}
			}
		}
	}
	if err = scanner.Err(); err != nil {
		t.Errorf("scanner encountered error: %v", err)
	}
	if found != len(funcs) {
		t.Error("missing functions")
	}
}

func TestIssue68411(t *testing.T) {
	globalSkip(t)
	testenv.MustHaveCGO(t)

	t.Parallel()

	// Test that the export header uses a void function parameter for
	// exported Go functions with no parameters.

	tmpdir := t.TempDir()

	const exportHeader = "issue68411.h"

	run(t, nil, "go", "tool", "cgo", "-exportheader", exportHeader, "-objdir", tmpdir, "./issue68411/issue68411.go")
	data, err := os.ReadFile(exportHeader)
	if err != nil {
		t.Fatal(err)
	}

	funcs := []struct{ name, signature string }{
		{"exportFuncWithNoParams", "void exportFuncWithNoParams(void)"},
		{"exportFuncWithParams", "exportFuncWithParams(GoInt a, GoInt b)"},
	}

	var found int
	for line := range bytes.Lines(data) {
		for _, fn := range funcs {
			if bytes.Contains(line, []byte(fn.name)) {
				found++
				if !bytes.Contains(line, []byte(fn.signature)) {
					t.Errorf("function signature mismatch; got %q, want %q", line, fn.signature)
				}
			}
		}
	}

	if found != len(funcs) {
		t.Error("missing functions")
	}
}
