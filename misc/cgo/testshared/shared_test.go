// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package shared_test

import (
	"bufio"
	"bytes"
	"debug/elf"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

var gopathInstallDir, gorootInstallDir, suffix string

// This is the smallest set of packages we can link into a shared
// library (runtime/cgo is built implicitly).
var minpkgs = []string{"runtime", "sync/atomic"}
var soname = "libruntime,sync-atomic.so"

// run runs a command and calls t.Errorf if it fails.
func run(t *testing.T, msg string, args ...string) {
	c := exec.Command(args[0], args[1:]...)
	if output, err := c.CombinedOutput(); err != nil {
		t.Errorf("executing %s (%s) failed %s:\n%s", strings.Join(args, " "), msg, err, output)
	}
}

// goCmd invokes the go tool with the installsuffix set up by TestMain. It calls
// t.Errorf if the command fails.
func goCmd(t *testing.T, args ...string) {
	newargs := []string{args[0], "-installsuffix=" + suffix}
	if testing.Verbose() {
		newargs = append(newargs, "-v")
	}
	newargs = append(newargs, args[1:]...)
	c := exec.Command("go", newargs...)
	if testing.Verbose() {
		fmt.Printf("+ go %s\n", strings.Join(newargs, " "))
	}
	if output, err := c.CombinedOutput(); err != nil {
		if t != nil {
			t.Errorf("executing %s failed %v:\n%s", strings.Join(c.Args, " "), err, output)
		} else {
			log.Fatalf("executing %s failed %v:\n%s", strings.Join(c.Args, " "), err, output)
		}
	}
}

// TestMain calls testMain so that the latter can use defer (TestMain exits with os.Exit).
func testMain(m *testing.M) (int, error) {
	// Because go install -buildmode=shared $standard_library_package always
	// installs into $GOROOT, here are some gymnastics to come up with a unique
	// installsuffix to use in this test that we can clean up afterwards.
	myContext := build.Default
	runtimeP, err := myContext.Import("runtime", ".", build.ImportComment)
	if err != nil {
		return 0, fmt.Errorf("import failed: %v", err)
	}
	for i := 0; i < 10000; i++ {
		try := fmt.Sprintf("%s_%d_dynlink", runtimeP.PkgTargetRoot, rand.Int63())
		err = os.Mkdir(try, 0700)
		if os.IsExist(err) {
			continue
		}
		if err == nil {
			gorootInstallDir = try
		}
		break
	}
	if err != nil {
		return 0, fmt.Errorf("can't create temporary directory: %v", err)
	}
	if gorootInstallDir == "" {
		return 0, errors.New("could not create temporary directory after 10000 tries")
	}
	defer os.RemoveAll(gorootInstallDir)

	// Some tests need to edit the source in GOPATH, so copy this directory to a
	// temporary directory and chdir to that.
	scratchDir, err := ioutil.TempDir("", "testshared")
	if err != nil {
		return 0, fmt.Errorf("TempDir failed: %v", err)
	}
	defer os.RemoveAll(scratchDir)
	err = filepath.Walk(".", func(path string, info os.FileInfo, err error) error {
		scratchPath := filepath.Join(scratchDir, path)
		if info.IsDir() {
			if path == "." {
				return nil
			}
			return os.Mkdir(scratchPath, info.Mode())
		} else {
			fromBytes, err := ioutil.ReadFile(path)
			if err != nil {
				return err
			}
			return ioutil.WriteFile(scratchPath, fromBytes, info.Mode())
		}
	})
	if err != nil {
		return 0, fmt.Errorf("walk failed: %v", err)
	}
	os.Setenv("GOPATH", scratchDir)
	myContext.GOPATH = scratchDir
	os.Chdir(scratchDir)

	// All tests depend on runtime being built into a shared library. Because
	// that takes a few seconds, do it here and have all tests use the version
	// built here.
	suffix = strings.Split(filepath.Base(gorootInstallDir), "_")[2]
	goCmd(nil, append([]string{"install", "-buildmode=shared"}, minpkgs...)...)

	myContext.InstallSuffix = suffix + "_dynlink"
	depP, err := myContext.Import("dep", ".", build.ImportComment)
	if err != nil {
		return 0, fmt.Errorf("import failed: %v", err)
	}
	gopathInstallDir = depP.PkgTargetRoot
	return m.Run(), nil
}

func TestMain(m *testing.M) {
	flag.Parse()
	exitCode, err := testMain(m)
	if err != nil {
		log.Fatal(err)
	}
	os.Exit(exitCode)
}

// The shared library was built at the expected location.
func TestSOBuilt(t *testing.T) {
	_, err := os.Stat(filepath.Join(gorootInstallDir, soname))
	if err != nil {
		t.Error(err)
	}
}

// The install command should have created a "shlibname" file for the
// listed packages (and runtime/cgo) indicating the name of the shared
// library containing it.
func TestShlibnameFiles(t *testing.T) {
	pkgs := append([]string{}, minpkgs...)
	pkgs = append(pkgs, "runtime/cgo")
	for _, pkg := range pkgs {
		shlibnamefile := filepath.Join(gorootInstallDir, pkg+".shlibname")
		contentsb, err := ioutil.ReadFile(shlibnamefile)
		if err != nil {
			t.Errorf("error reading shlibnamefile for %s: %v", pkg, err)
			continue
		}
		contents := strings.TrimSpace(string(contentsb))
		if contents != soname {
			t.Errorf("shlibnamefile for %s has wrong contents: %q", pkg, contents)
		}
	}
}

func dynStrings(path string, flag elf.DynTag) []string {
	f, err := elf.Open(path)
	defer f.Close()
	if err != nil {
		log.Fatal("elf.Open failed: ", err)
	}
	dynstrings, err := f.DynString(flag)
	if err != nil {
		log.Fatal("dynstring failed: ", err)
	}
	return dynstrings
}

func AssertIsLinkedTo(t *testing.T, path, lib string) {
	for _, dynstring := range dynStrings(path, elf.DT_NEEDED) {
		if dynstring == lib {
			return
		}
	}
	t.Errorf("%s is not linked to %s", path, lib)
}

func AssertHasRPath(t *testing.T, path, dir string) {
	for _, tag := range []elf.DynTag{elf.DT_RPATH, elf.DT_RUNPATH} {
		for _, dynstring := range dynStrings(path, tag) {
			for _, rpath := range strings.Split(dynstring, ":") {
				if filepath.Clean(rpath) == filepath.Clean(dir) {
					return
				}
			}
		}
	}
	t.Errorf("%s does not have rpath %s", path, dir)
}

// Build a trivial program that links against the shared runtime and check it runs.
func TestTrivialExecutable(t *testing.T) {
	goCmd(t, "install", "-linkshared", "trivial")
	run(t, "trivial executable", "./bin/trivial")
	AssertIsLinkedTo(t, "./bin/trivial", soname)
	AssertHasRPath(t, "./bin/trivial", gorootInstallDir)
}

// Build a GOPATH package into a shared library that links against the goroot runtime
// and an executable that links against both.
func TestGOPathShlib(t *testing.T) {
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "dep")
	AssertIsLinkedTo(t, filepath.Join(gopathInstallDir, "libdep.so"), soname)
	goCmd(t, "install", "-linkshared", "exe")
	AssertIsLinkedTo(t, "./bin/exe", soname)
	AssertIsLinkedTo(t, "./bin/exe", "libdep.so")
	AssertHasRPath(t, "./bin/exe", gorootInstallDir)
	AssertHasRPath(t, "./bin/exe", gopathInstallDir)
	// And check it runs.
	run(t, "executable linked to GOPATH library", "./bin/exe")
}

// Testing rebuilding of shared libraries when they are stale is a bit more
// complicated that it seems like it should be. First, we make everything "old": but
// only a few seconds old, or it might be older than 6g (or the runtime source) and
// everything will get rebuilt. Then define a timestamp slightly newer than this
// time, which is what we set the mtime to of a file to cause it to be seen as new,
// and finally another slightly even newer one that we can compare files against to
// see if they have been rebuilt.
var oldTime = time.Now().Add(-9 * time.Second)
var nearlyNew = time.Now().Add(-6 * time.Second)
var stampTime = time.Now().Add(-3 * time.Second)

// resetFileStamps makes "everything" (bin, src, pkg from GOPATH and the
// test-specific parts of GOROOT) appear old.
func resetFileStamps() {
	chtime := func(path string, info os.FileInfo, err error) error {
		return os.Chtimes(path, oldTime, oldTime)
	}
	reset := func(path string) {
		if err := filepath.Walk(path, chtime); err != nil {
			log.Fatalf("resetFileStamps failed: %v", err)
		}

	}
	reset("bin")
	reset("pkg")
	reset("src")
	reset(gorootInstallDir)
}

// touch makes path newer than the "old" time stamp used by resetFileStamps.
func touch(path string) {
	if err := os.Chtimes(path, nearlyNew, nearlyNew); err != nil {
		log.Fatalf("os.Chtimes failed: %v", err)
	}
}

// isNew returns if the path is newer than the time stamp used by touch.
func isNew(path string) bool {
	fi, err := os.Stat(path)
	if err != nil {
		log.Fatalf("os.Stat failed: %v", err)
	}
	return fi.ModTime().After(stampTime)
}

// Fail unless path has been rebuilt (i.e. is newer than the time stamp used by
// isNew)
func AssertRebuilt(t *testing.T, msg, path string) {
	if !isNew(path) {
		t.Errorf("%s was not rebuilt (%s)", msg, path)
	}
}

// Fail if path has been rebuilt (i.e. is newer than the time stamp used by isNew)
func AssertNotRebuilt(t *testing.T, msg, path string) {
	if isNew(path) {
		t.Errorf("%s was rebuilt (%s)", msg, path)
	}
}

func TestRebuilding(t *testing.T) {
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "dep")
	goCmd(t, "install", "-linkshared", "exe")

	// If the source is newer than both the .a file and the .so, both are rebuilt.
	resetFileStamps()
	touch("src/dep/dep.go")
	goCmd(t, "install", "-linkshared", "exe")
	AssertRebuilt(t, "new source", filepath.Join(gopathInstallDir, "dep.a"))
	AssertRebuilt(t, "new source", filepath.Join(gopathInstallDir, "libdep.so"))

	// If the .a file is newer than the .so, the .so is rebuilt (but not the .a)
	resetFileStamps()
	touch(filepath.Join(gopathInstallDir, "dep.a"))
	goCmd(t, "install", "-linkshared", "exe")
	AssertNotRebuilt(t, "new .a file", filepath.Join(gopathInstallDir, "dep.a"))
	AssertRebuilt(t, "new .a file", filepath.Join(gopathInstallDir, "libdep.so"))
}

func appendFile(path, content string) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0660)
	if err != nil {
		log.Fatalf("os.OpenFile failed: %v", err)
	}
	defer func() {
		err := f.Close()
		if err != nil {
			log.Fatalf("f.Close failed: %v", err)
		}
	}()
	_, err = f.WriteString(content)
	if err != nil {
		log.Fatalf("f.WriteString failed: %v", err)
	}
}

func TestABIChecking(t *testing.T) {
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "dep")
	goCmd(t, "install", "-linkshared", "exe")

	// If we make an ABI-breaking change to dep and rebuild libp.so but not exe,
	// exe will abort with a complaint on startup.
	// This assumes adding an exported function breaks ABI, which is not true in
	// some senses but suffices for the narrow definition of ABI compatiblity the
	// toolchain uses today.
	appendFile("src/dep/dep.go", "func ABIBreak() {}\n")
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "dep")
	c := exec.Command("./bin/exe")
	output, err := c.CombinedOutput()
	if err == nil {
		t.Fatal("executing exe did not fail after ABI break")
	}
	scanner := bufio.NewScanner(bytes.NewReader(output))
	foundMsg := false
	const wantLine = "abi mismatch detected between the executable and libdep.so"
	for scanner.Scan() {
		if scanner.Text() == wantLine {
			foundMsg = true
			break
		}
	}
	if err = scanner.Err(); err != nil {
		t.Errorf("scanner encountered error: %v", err)
	}
	if !foundMsg {
		t.Fatalf("exe failed, but without line %q; got output:\n%s", wantLine, output)
	}

	// Rebuilding exe makes it work again.
	goCmd(t, "install", "-linkshared", "exe")
	run(t, "rebuilt exe", "./bin/exe")

	// If we make a change which does not break ABI (such as adding an unexported
	// function) and rebuild libdep.so, exe still works.
	appendFile("src/dep/dep.go", "func noABIBreak() {}\n")
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "dep")
	run(t, "after non-ABI breaking change", "./bin/exe")
}
