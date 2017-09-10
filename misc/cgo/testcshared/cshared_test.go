// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cshared_test

import (
	"debug/elf"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"unicode"
)

// C compiler with args (from $(go env CC) $(go env GOGCCFLAGS)).
var cc []string

// An environment with GOPATH=$(pwd).
var gopathEnv []string

// ".exe" on Windows.
var exeSuffix string

var GOOS, GOARCH, GOROOT string
var installdir, androiddir string
var libSuffix, libgoname string

func TestMain(m *testing.M) {
	GOOS = goEnv("GOOS")
	GOARCH = goEnv("GOARCH")
	GOROOT = goEnv("GOROOT")

	if _, err := os.Stat(GOROOT); os.IsNotExist(err) {
		log.Fatalf("Unable able to find GOROOT at '%s'", GOROOT)
	}

	// Directory where cgo headers and outputs will be installed.
	// The installation directory format varies depending on the platform.
	installdir = path.Join("pkg", fmt.Sprintf("%s_%s_testcshared_shared", GOOS, GOARCH))
	switch GOOS {
	case "darwin":
		libSuffix = "dylib"
		installdir = path.Join("pkg", fmt.Sprintf("%s_%s_testcshared", GOOS, GOARCH))
	case "windows":
		libSuffix = "dll"
	default:
		libSuffix = "so"
	}

	androiddir = fmt.Sprintf("/data/local/tmp/testcshared-%d", os.Getpid())
	if GOOS == "android" {
		cmd := exec.Command("adb", "shell", "mkdir", "-p", androiddir)
		out, err := cmd.CombinedOutput()
		if err != nil {
			log.Fatalf("setupAndroid failed: %v\n%s\n", err, out)
		}
	}

	libgoname = "libgo." + libSuffix

	ccOut := goEnv("CC")
	cc = []string{string(ccOut)}

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
	case "darwin":
		// For Darwin/ARM.
		// TODO(crawshaw): can we do better?
		cc = append(cc, []string{"-framework", "CoreFoundation", "-framework", "Foundation"}...)
	case "android":
		cc = append(cc, "-pie", "-fuse-ld=gold")
	}
	libgodir := GOOS + "_" + GOARCH
	switch GOOS {
	case "darwin":
		if GOARCH == "arm" || GOARCH == "arm64" {
			libgodir += "_shared"
		}
	case "dragonfly", "freebsd", "linux", "netbsd", "openbsd", "solaris":
		libgodir += "_shared"
	}
	cc = append(cc, "-I", filepath.Join("pkg", libgodir))

	// Build an environment with GOPATH=$(pwd)
	dir, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	gopathEnv = append(os.Environ(), "GOPATH="+dir)

	if GOOS == "windows" {
		exeSuffix = ".exe"
	}

	st := m.Run()

	os.Remove(libgoname)
	os.RemoveAll("pkg")
	cleanupHeaders()
	cleanupAndroid()

	os.Exit(st)
}

func goEnv(key string) string {
	out, err := exec.Command("go", "env", key).Output()
	if err != nil {
		fmt.Fprintf(os.Stderr, "go env %s failed:\n%s", key, err)
		fmt.Fprintf(os.Stderr, "%s", err.(*exec.ExitError).Stderr)
		os.Exit(2)
	}
	return strings.TrimSpace(string(out))
}

func cmdToRun(name string) []string {
	return []string{"./" + name + exeSuffix}
}

func adbPush(t *testing.T, filename string) {
	if GOOS != "android" {
		return
	}
	args := []string{"adb", "push", filename, fmt.Sprintf("%s/%s", androiddir, filename)}
	cmd := exec.Command(args[0], args[1:]...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("adb command failed: %v\n%s\n", err, out)
	}
}

func adbRun(t *testing.T, env []string, adbargs ...string) string {
	if GOOS != "android" {
		t.Fatalf("trying to run adb command when operating system is not android.")
	}
	args := []string{"adb", "shell"}
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

func runwithenv(t *testing.T, env []string, args ...string) string {
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Env = env
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("command failed: %v\n%v\n%s\n", args, err, out)
	} else {
		t.Logf("run: %v", args)
	}

	return string(out)
}

func run(t *testing.T, args ...string) string {
	cmd := exec.Command(args[0], args[1:]...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("command failed: %v\n%v\n%s\n", args, err, out)
	} else {
		t.Logf("run: %v", args)
	}

	return string(out)
}

func runExe(t *testing.T, env []string, args ...string) string {
	if GOOS == "android" {
		return adbRun(t, env, args...)
	}

	return runwithenv(t, env, args...)
}

func runwithldlibrarypath(t *testing.T, args ...string) string {
	return runExe(t, append(gopathEnv, "LD_LIBRARY_PATH=."), args...)
}

func rungocmd(t *testing.T, args ...string) string {
	return runwithenv(t, gopathEnv, args...)
}

func createHeaders() error {
	args := []string{"go", "install", "-buildmode=c-shared",
		"-installsuffix", "testcshared", "libgo"}
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Env = gopathEnv
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("command failed: %v\n%v\n%s\n", args, err, out)
	}

	args = []string{"go", "build", "-buildmode=c-shared",
		"-installsuffix", "testcshared",
		"-o", libgoname,
		filepath.Join("src", "libgo", "libgo.go")}
	cmd = exec.Command(args[0], args[1:]...)
	cmd.Env = gopathEnv
	out, err = cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("command failed: %v\n%v\n%s\n", args, err, out)
	}

	if GOOS == "android" {
		args = []string{"adb", "push", libgoname, fmt.Sprintf("%s/%s", androiddir, libgoname)}
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

func cleanupHeaders() {
	os.Remove("libgo.h")
}

func cleanupAndroid() {
	if GOOS != "android" {
		return
	}
	cmd := exec.Command("adb", "shell", "rm", "-rf", androiddir)
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatalf("cleanupAndroid failed: %v\n%s\n", err, out)
	}
}

// test0: exported symbols in shared lib are accessible.
func TestExportedSymbols(t *testing.T) {
	t.Parallel()

	cmd := "testp0"
	bin := cmdToRun(cmd)

	createHeadersOnce(t)

	run(t, append(cc, "-I", installdir, "-o", cmd, "main0.c", libgoname)...)
	adbPush(t, cmd)

	defer os.Remove(cmd)

	out := runwithldlibrarypath(t, bin...)
	if strings.TrimSpace(out) != "PASS" {
		t.Error(out)
	}
}

// test1: shared library can be dynamically loaded and exported symbols are accessible.
func TestExportedSymbolsWithDynamicLoad(t *testing.T) {
	t.Parallel()

	cmd := "testp1"
	bin := cmdToRun(cmd)

	createHeadersOnce(t)

	run(t, append(cc, "-o", cmd, "main1.c", "-ldl")...)
	adbPush(t, cmd)

	defer os.Remove(cmd)

	out := runExe(t, nil, append(bin, "./"+libgoname)...)
	if strings.TrimSpace(out) != "PASS" {
		t.Error(out)
	}
}

// test2: tests libgo2 which does not export any functions.
func TestUnexportedSymbols(t *testing.T) {
	t.Parallel()

	cmd := "testp2"
	libname := "libgo2." + libSuffix
	bin := cmdToRun(cmd)

	rungocmd(t,
		"go", "build",
		"-buildmode=c-shared",
		"-installsuffix", "testcshared",
		"-o", libname, "libgo2",
	)
	adbPush(t, libname)

	linkFlags := "-Wl,--no-as-needed"
	if GOOS == "darwin" {
		linkFlags = ""
	}

	run(t, append(
		cc, "-o", cmd,
		"main2.c", linkFlags,
		libname,
	)...)
	adbPush(t, cmd)

	defer os.Remove(libname)
	defer os.Remove(cmd)

	out := runwithldlibrarypath(t, bin...)

	if strings.TrimSpace(out) != "PASS" {
		t.Error(out)
	}
}

// test3: tests main.main is exported on android.
func TestMainExportedOnAndroid(t *testing.T) {
	t.Parallel()

	if GOOS != "android" {
		return
	}

	cmd := "testp3"
	bin := cmdToRun(cmd)

	createHeadersOnce(t)

	run(t, append(cc, "-o", cmd, "main3.c", "-ldl")...)
	adbPush(t, cmd)

	defer os.Remove(cmd)

	out := runExe(t, nil, append(bin, "./"+libgoname)...)
	if strings.TrimSpace(out) != "PASS" {
		t.Error(out)
	}
}

// test4: test signal handlers
func TestSignalHandlers(t *testing.T) {
	t.Parallel()

	cmd := "testp4"
	libname := "libgo4." + libSuffix
	bin := cmdToRun(cmd)

	rungocmd(t,
		"go", "build",
		"-buildmode=c-shared",
		"-installsuffix", "testcshared",
		"-o", libname, "libgo4",
	)
	adbPush(t, libname)
	run(t, append(
		cc, "-pthread", "-o", cmd,
		"main4.c", "-ldl",
	)...)
	adbPush(t, cmd)

	defer os.Remove(libname)
	defer os.Remove(cmd)
	defer os.Remove("libgo4.h")

	out := runExe(t, nil, append(bin, "./"+libname)...)

	if strings.TrimSpace(out) != "PASS" {
		t.Error(run(t, append(bin, libname, "verbose")...))
	}
}

// test5: test signal handlers with os/signal.Notify
func TestSignalHandlersWithNotify(t *testing.T) {
	t.Parallel()

	cmd := "testp5"
	libname := "libgo5." + libSuffix
	bin := cmdToRun(cmd)

	rungocmd(t,
		"go", "build",
		"-buildmode=c-shared",
		"-installsuffix", "testcshared",
		"-o", libname, "libgo5",
	)
	adbPush(t, libname)
	run(t, append(
		cc, "-pthread", "-o", cmd,
		"main5.c", "-ldl",
	)...)
	adbPush(t, cmd)

	defer os.Remove(libname)
	defer os.Remove(cmd)
	defer os.Remove("libgo5.h")

	out := runExe(t, nil, append(bin, "./"+libname)...)

	if strings.TrimSpace(out) != "PASS" {
		t.Error(run(t, append(bin, libname, "verbose")...))
	}
}

func TestPIE(t *testing.T) {
	t.Parallel()

	switch GOOS {
	case "linux", "android":
		break
	default:
		t.Logf("Skipping TestPIE on %s", GOOS)
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
