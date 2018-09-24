// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package shared_test

import (
	"bufio"
	"bytes"
	"debug/elf"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
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
// t.Fatalf if the command fails.
func goCmd(t *testing.T, args ...string) {
	newargs := []string{args[0], "-installsuffix=" + suffix}
	if testing.Verbose() {
		newargs = append(newargs, "-x")
	}
	newargs = append(newargs, args[1:]...)
	c := exec.Command("go", newargs...)
	var output []byte
	var err error
	if testing.Verbose() {
		fmt.Printf("+ go %s\n", strings.Join(newargs, " "))
		c.Stdout = os.Stdout
		c.Stderr = os.Stderr
		err = c.Run()
		output = []byte("(output above)")
	} else {
		output, err = c.CombinedOutput()
	}
	if err != nil {
		if t != nil {
			t.Fatalf("executing %s failed %v:\n%s", strings.Join(c.Args, " "), err, output)
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
	if testing.Verbose() {
		fmt.Printf("+ mkdir -p %s\n", gorootInstallDir)
	}
	defer os.RemoveAll(gorootInstallDir)

	// Some tests need to edit the source in GOPATH, so copy this directory to a
	// temporary directory and chdir to that.
	scratchDir, err := ioutil.TempDir("", "testshared")
	if err != nil {
		return 0, fmt.Errorf("TempDir failed: %v", err)
	}
	if testing.Verbose() {
		fmt.Printf("+ mkdir -p %s\n", scratchDir)
	}
	defer os.RemoveAll(scratchDir)
	err = filepath.Walk(".", func(path string, info os.FileInfo, err error) error {
		scratchPath := filepath.Join(scratchDir, path)
		if info.IsDir() {
			if path == "." {
				return nil
			}
			if testing.Verbose() {
				fmt.Printf("+ mkdir -p %s\n", scratchPath)
			}
			return os.Mkdir(scratchPath, info.Mode())
		} else {
			fromBytes, err := ioutil.ReadFile(path)
			if err != nil {
				return err
			}
			if testing.Verbose() {
				fmt.Printf("+ cp %s %s\n", path, scratchPath)
			}
			return ioutil.WriteFile(scratchPath, fromBytes, info.Mode())
		}
	})
	if err != nil {
		return 0, fmt.Errorf("walk failed: %v", err)
	}
	os.Setenv("GOPATH", scratchDir)
	if testing.Verbose() {
		fmt.Printf("+ export GOPATH=%s\n", scratchDir)
	}
	myContext.GOPATH = scratchDir
	if testing.Verbose() {
		fmt.Printf("+ cd %s\n", scratchDir)
	}
	os.Chdir(scratchDir)

	// All tests depend on runtime being built into a shared library. Because
	// that takes a few seconds, do it here and have all tests use the version
	// built here.
	suffix = strings.Split(filepath.Base(gorootInstallDir), "_")[2]
	goCmd(nil, append([]string{"install", "-buildmode=shared"}, minpkgs...)...)

	myContext.InstallSuffix = suffix + "_dynlink"
	depP, err := myContext.Import("depBase", ".", build.ImportComment)
	if err != nil {
		return 0, fmt.Errorf("import failed: %v", err)
	}
	gopathInstallDir = depP.PkgTargetRoot
	return m.Run(), nil
}

func TestMain(m *testing.M) {
	flag.Parse()

	// Some of the tests install binaries into a custom GOPATH.
	// That won't work if GOBIN is set.
	os.Unsetenv("GOBIN")

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

func hasDynTag(f *elf.File, tag elf.DynTag) bool {
	ds := f.SectionByType(elf.SHT_DYNAMIC)
	if ds == nil {
		return false
	}
	d, err := ds.Data()
	if err != nil {
		return false
	}
	for len(d) > 0 {
		var t elf.DynTag
		switch f.Class {
		case elf.ELFCLASS32:
			t = elf.DynTag(f.ByteOrder.Uint32(d[0:4]))
			d = d[8:]
		case elf.ELFCLASS64:
			t = elf.DynTag(f.ByteOrder.Uint64(d[0:8]))
			d = d[16:]
		}
		if t == tag {
			return true
		}
	}
	return false
}

// The shared library does not have relocations against the text segment.
func TestNoTextrel(t *testing.T) {
	sopath := filepath.Join(gorootInstallDir, soname)
	f, err := elf.Open(sopath)
	if err != nil {
		t.Fatal("elf.Open failed: ", err)
	}
	defer f.Close()
	if hasDynTag(f, elf.DT_TEXTREL) {
		t.Errorf("%s has DT_TEXTREL set", soname)
	}
}

// The shared library does not contain symbols called ".dup"
func TestNoDupSymbols(t *testing.T) {
	sopath := filepath.Join(gorootInstallDir, soname)
	f, err := elf.Open(sopath)
	if err != nil {
		t.Fatal("elf.Open failed: ", err)
	}
	defer f.Close()
	syms, err := f.Symbols()
	if err != nil {
		t.Errorf("error reading symbols %v", err)
		return
	}
	for _, s := range syms {
		if s.Name == ".dup" {
			t.Fatalf("%s contains symbol called .dup", sopath)
		}
	}
}

// The install command should have created a "shlibname" file for the
// listed packages (and runtime/cgo, and math on arm) indicating the
// name of the shared library containing it.
func TestShlibnameFiles(t *testing.T) {
	pkgs := append([]string{}, minpkgs...)
	pkgs = append(pkgs, "runtime/cgo")
	if runtime.GOARCH == "arm" {
		pkgs = append(pkgs, "math")
	}
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

// Is a given offset into the file contained in a loaded segment?
func isOffsetLoaded(f *elf.File, offset uint64) bool {
	for _, prog := range f.Progs {
		if prog.Type == elf.PT_LOAD {
			if prog.Off <= offset && offset < prog.Off+prog.Filesz {
				return true
			}
		}
	}
	return false
}

func rnd(v int32, r int32) int32 {
	if r <= 0 {
		return v
	}
	v += r - 1
	c := v % r
	if c < 0 {
		c += r
	}
	v -= c
	return v
}

func readwithpad(r io.Reader, sz int32) ([]byte, error) {
	data := make([]byte, rnd(sz, 4))
	_, err := io.ReadFull(r, data)
	if err != nil {
		return nil, err
	}
	data = data[:sz]
	return data, nil
}

type note struct {
	name    string
	tag     int32
	desc    string
	section *elf.Section
}

// Read all notes from f. As ELF section names are not supposed to be special, one
// looks for a particular note by scanning all SHT_NOTE sections looking for a note
// with a particular "name" and "tag".
func readNotes(f *elf.File) ([]*note, error) {
	var notes []*note
	for _, sect := range f.Sections {
		if sect.Type != elf.SHT_NOTE {
			continue
		}
		r := sect.Open()
		for {
			var namesize, descsize, tag int32
			err := binary.Read(r, f.ByteOrder, &namesize)
			if err != nil {
				if err == io.EOF {
					break
				}
				return nil, fmt.Errorf("read namesize failed: %v", err)
			}
			err = binary.Read(r, f.ByteOrder, &descsize)
			if err != nil {
				return nil, fmt.Errorf("read descsize failed: %v", err)
			}
			err = binary.Read(r, f.ByteOrder, &tag)
			if err != nil {
				return nil, fmt.Errorf("read type failed: %v", err)
			}
			name, err := readwithpad(r, namesize)
			if err != nil {
				return nil, fmt.Errorf("read name failed: %v", err)
			}
			desc, err := readwithpad(r, descsize)
			if err != nil {
				return nil, fmt.Errorf("read desc failed: %v", err)
			}
			notes = append(notes, &note{name: string(name), tag: tag, desc: string(desc), section: sect})
		}
	}
	return notes, nil
}

func dynStrings(t *testing.T, path string, flag elf.DynTag) []string {
	f, err := elf.Open(path)
	if err != nil {
		t.Fatalf("elf.Open(%q) failed: %v", path, err)
	}
	defer f.Close()
	dynstrings, err := f.DynString(flag)
	if err != nil {
		t.Fatalf("DynString(%s) failed on %s: %v", flag, path, err)
	}
	return dynstrings
}

func AssertIsLinkedToRegexp(t *testing.T, path string, re *regexp.Regexp) {
	for _, dynstring := range dynStrings(t, path, elf.DT_NEEDED) {
		if re.MatchString(dynstring) {
			return
		}
	}
	t.Errorf("%s is not linked to anything matching %v", path, re)
}

func AssertIsLinkedTo(t *testing.T, path, lib string) {
	AssertIsLinkedToRegexp(t, path, regexp.MustCompile(regexp.QuoteMeta(lib)))
}

func AssertHasRPath(t *testing.T, path, dir string) {
	for _, tag := range []elf.DynTag{elf.DT_RPATH, elf.DT_RUNPATH} {
		for _, dynstring := range dynStrings(t, path, tag) {
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

// Build a trivial program in PIE mode that links against the shared runtime and check it runs.
func TestTrivialExecutablePIE(t *testing.T) {
	goCmd(t, "build", "-buildmode=pie", "-o", "trivial.pie", "-linkshared", "trivial")
	run(t, "trivial executable", "./trivial.pie")
	AssertIsLinkedTo(t, "./trivial.pie", soname)
	AssertHasRPath(t, "./trivial.pie", gorootInstallDir)
}

// Build a division test program and check it runs.
func TestDivisionExecutable(t *testing.T) {
	goCmd(t, "install", "-linkshared", "division")
	run(t, "division executable", "./bin/division")
}

// Build an executable that uses cgo linked against the shared runtime and check it
// runs.
func TestCgoExecutable(t *testing.T) {
	goCmd(t, "install", "-linkshared", "execgo")
	run(t, "cgo executable", "./bin/execgo")
}

func checkPIE(t *testing.T, name string) {
	f, err := elf.Open(name)
	if err != nil {
		t.Fatal("elf.Open failed: ", err)
	}
	defer f.Close()
	if f.Type != elf.ET_DYN {
		t.Errorf("%s has type %v, want ET_DYN", name, f.Type)
	}
	if hasDynTag(f, elf.DT_TEXTREL) {
		t.Errorf("%s has DT_TEXTREL set", name)
	}
}

func TestTrivialPIE(t *testing.T) {
	name := "trivial_pie"
	goCmd(t, "build", "-buildmode=pie", "-o="+name, "trivial")
	defer os.Remove(name)
	run(t, name, "./"+name)
	checkPIE(t, name)
}

func TestCgoPIE(t *testing.T) {
	name := "cgo_pie"
	goCmd(t, "build", "-buildmode=pie", "-o="+name, "execgo")
	defer os.Remove(name)
	run(t, name, "./"+name)
	checkPIE(t, name)
}

// Build a GOPATH package into a shared library that links against the goroot runtime
// and an executable that links against both.
func TestGopathShlib(t *testing.T) {
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "depBase")
	AssertIsLinkedTo(t, filepath.Join(gopathInstallDir, "libdepBase.so"), soname)
	goCmd(t, "install", "-linkshared", "exe")
	AssertIsLinkedTo(t, "./bin/exe", soname)
	AssertIsLinkedTo(t, "./bin/exe", "libdepBase.so")
	AssertHasRPath(t, "./bin/exe", gorootInstallDir)
	AssertHasRPath(t, "./bin/exe", gopathInstallDir)
	// And check it runs.
	run(t, "executable linked to GOPATH library", "./bin/exe")
}

// The shared library contains a note listing the packages it contains in a section
// that is not mapped into memory.
func testPkgListNote(t *testing.T, f *elf.File, note *note) {
	if note.section.Flags != 0 {
		t.Errorf("package list section has flags %v, want 0", note.section.Flags)
	}
	if isOffsetLoaded(f, note.section.Offset) {
		t.Errorf("package list section contained in PT_LOAD segment")
	}
	if note.desc != "depBase\n" {
		t.Errorf("incorrect package list %q, want %q", note.desc, "depBase\n")
	}
}

// The shared library contains a note containing the ABI hash that is mapped into
// memory and there is a local symbol called go.link.abihashbytes that points 16
// bytes into it.
func testABIHashNote(t *testing.T, f *elf.File, note *note) {
	if note.section.Flags != elf.SHF_ALLOC {
		t.Errorf("abi hash section has flags %v, want SHF_ALLOC", note.section.Flags)
	}
	if !isOffsetLoaded(f, note.section.Offset) {
		t.Errorf("abihash section not contained in PT_LOAD segment")
	}
	var hashbytes elf.Symbol
	symbols, err := f.Symbols()
	if err != nil {
		t.Errorf("error reading symbols %v", err)
		return
	}
	for _, sym := range symbols {
		if sym.Name == "go.link.abihashbytes" {
			hashbytes = sym
		}
	}
	if hashbytes.Name == "" {
		t.Errorf("no symbol called go.link.abihashbytes")
		return
	}
	if elf.ST_BIND(hashbytes.Info) != elf.STB_LOCAL {
		t.Errorf("%s has incorrect binding %v, want STB_LOCAL", hashbytes.Name, elf.ST_BIND(hashbytes.Info))
	}
	if f.Sections[hashbytes.Section] != note.section {
		t.Errorf("%s has incorrect section %v, want %s", hashbytes.Name, f.Sections[hashbytes.Section].Name, note.section.Name)
	}
	if hashbytes.Value-note.section.Addr != 16 {
		t.Errorf("%s has incorrect offset into section %d, want 16", hashbytes.Name, hashbytes.Value-note.section.Addr)
	}
}

// A Go shared library contains a note indicating which other Go shared libraries it
// was linked against in an unmapped section.
func testDepsNote(t *testing.T, f *elf.File, note *note) {
	if note.section.Flags != 0 {
		t.Errorf("package list section has flags %v, want 0", note.section.Flags)
	}
	if isOffsetLoaded(f, note.section.Offset) {
		t.Errorf("package list section contained in PT_LOAD segment")
	}
	// libdepBase.so just links against the lib containing the runtime.
	if note.desc != soname {
		t.Errorf("incorrect dependency list %q, want %q", note.desc, soname)
	}
}

// The shared library contains notes with defined contents; see above.
func TestNotes(t *testing.T) {
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "depBase")
	f, err := elf.Open(filepath.Join(gopathInstallDir, "libdepBase.so"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	notes, err := readNotes(f)
	if err != nil {
		t.Fatal(err)
	}
	pkgListNoteFound := false
	abiHashNoteFound := false
	depsNoteFound := false
	for _, note := range notes {
		if note.name != "Go\x00\x00" {
			continue
		}
		switch note.tag {
		case 1: // ELF_NOTE_GOPKGLIST_TAG
			if pkgListNoteFound {
				t.Error("multiple package list notes")
			}
			testPkgListNote(t, f, note)
			pkgListNoteFound = true
		case 2: // ELF_NOTE_GOABIHASH_TAG
			if abiHashNoteFound {
				t.Error("multiple abi hash notes")
			}
			testABIHashNote(t, f, note)
			abiHashNoteFound = true
		case 3: // ELF_NOTE_GODEPS_TAG
			if depsNoteFound {
				t.Error("multiple depedency list notes")
			}
			testDepsNote(t, f, note)
			depsNoteFound = true
		}
	}
	if !pkgListNoteFound {
		t.Error("package list note not found")
	}
	if !abiHashNoteFound {
		t.Error("abi hash note not found")
	}
	if !depsNoteFound {
		t.Error("deps note not found")
	}
}

// Build a GOPATH package (depBase) into a shared library that links against the goroot
// runtime, another package (dep2) that links against the first, and and an
// executable that links against dep2.
func TestTwoGopathShlibs(t *testing.T) {
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "depBase")
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "dep2")
	goCmd(t, "install", "-linkshared", "exe2")
	run(t, "executable linked to GOPATH library", "./bin/exe2")
}

func TestThreeGopathShlibs(t *testing.T) {
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "depBase")
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "dep2")
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "dep3")
	goCmd(t, "install", "-linkshared", "exe3")
	run(t, "executable linked to GOPATH library", "./bin/exe3")
}

// If gccgo is not available or not new enough call t.Skip. Otherwise,
// return a build.Context that is set up for gccgo.
func prepGccgo(t *testing.T) build.Context {
	gccgoName := os.Getenv("GCCGO")
	if gccgoName == "" {
		gccgoName = "gccgo"
	}
	gccgoPath, err := exec.LookPath(gccgoName)
	if err != nil {
		t.Skip("gccgo not found")
	}
	cmd := exec.Command(gccgoPath, "-dumpversion")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s -dumpversion failed: %v\n%s", gccgoPath, err, output)
	}
	if string(output) < "5" {
		t.Skipf("gccgo too old (%s)", strings.TrimSpace(string(output)))
	}
	gccgoContext := build.Default
	gccgoContext.InstallSuffix = suffix + "_fPIC"
	gccgoContext.Compiler = "gccgo"
	gccgoContext.GOPATH = os.Getenv("GOPATH")
	return gccgoContext
}

// Build a GOPATH package into a shared library with gccgo and an executable that
// links against it.
func TestGoPathShlibGccgo(t *testing.T) {
	gccgoContext := prepGccgo(t)

	libgoRE := regexp.MustCompile("libgo.so.[0-9]+")

	depP, err := gccgoContext.Import("depBase", ".", build.ImportComment)
	if err != nil {
		t.Fatalf("import failed: %v", err)
	}
	gccgoInstallDir := filepath.Join(depP.PkgTargetRoot, "shlibs")
	goCmd(t, "install", "-compiler=gccgo", "-buildmode=shared", "-linkshared", "depBase")
	AssertIsLinkedToRegexp(t, filepath.Join(gccgoInstallDir, "libdepBase.so"), libgoRE)
	goCmd(t, "install", "-compiler=gccgo", "-linkshared", "exe")
	AssertIsLinkedToRegexp(t, "./bin/exe", libgoRE)
	AssertIsLinkedTo(t, "./bin/exe", "libdepBase.so")
	AssertHasRPath(t, "./bin/exe", gccgoInstallDir)
	// And check it runs.
	run(t, "gccgo-built", "./bin/exe")
}

// The gccgo version of TestTwoGopathShlibs: build a GOPATH package into a shared
// library with gccgo, another GOPATH package that depends on the first and an
// executable that links the second library.
func TestTwoGopathShlibsGccgo(t *testing.T) {
	gccgoContext := prepGccgo(t)

	libgoRE := regexp.MustCompile("libgo.so.[0-9]+")

	depP, err := gccgoContext.Import("depBase", ".", build.ImportComment)
	if err != nil {
		t.Fatalf("import failed: %v", err)
	}
	gccgoInstallDir := filepath.Join(depP.PkgTargetRoot, "shlibs")
	goCmd(t, "install", "-compiler=gccgo", "-buildmode=shared", "-linkshared", "depBase")
	goCmd(t, "install", "-compiler=gccgo", "-buildmode=shared", "-linkshared", "dep2")
	goCmd(t, "install", "-compiler=gccgo", "-linkshared", "exe2")

	AssertIsLinkedToRegexp(t, filepath.Join(gccgoInstallDir, "libdepBase.so"), libgoRE)
	AssertIsLinkedToRegexp(t, filepath.Join(gccgoInstallDir, "libdep2.so"), libgoRE)
	AssertIsLinkedTo(t, filepath.Join(gccgoInstallDir, "libdep2.so"), "libdepBase.so")
	AssertIsLinkedToRegexp(t, "./bin/exe2", libgoRE)
	AssertIsLinkedTo(t, "./bin/exe2", "libdep2")
	AssertIsLinkedTo(t, "./bin/exe2", "libdepBase.so")

	// And check it runs.
	run(t, "gccgo-built", "./bin/exe2")
}

// Testing rebuilding of shared libraries when they are stale is a bit more
// complicated that it seems like it should be. First, we make everything "old": but
// only a few seconds old, or it might be older than gc (or the runtime source) and
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

// touch changes path and returns a function that changes it back.
// It also sets the time of the file, so that we can see if it is rewritten.
func touch(t *testing.T, path string) (cleanup func()) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	old := make([]byte, len(data))
	copy(old, data)
	if bytes.HasPrefix(data, []byte("!<arch>\n")) {
		// Change last digit of build ID.
		// (Content ID in the new content-based build IDs.)
		const marker = `build id "`
		i := bytes.Index(data, []byte(marker))
		if i < 0 {
			t.Fatal("cannot find build id in archive")
		}
		j := bytes.IndexByte(data[i+len(marker):], '"')
		if j < 0 {
			t.Fatal("cannot find build id in archive")
		}
		i += len(marker) + j - 1
		if data[i] == 'a' {
			data[i] = 'b'
		} else {
			data[i] = 'a'
		}
	} else {
		// assume it's a text file
		data = append(data, '\n')
	}
	if err := ioutil.WriteFile(path, data, 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, nearlyNew, nearlyNew); err != nil {
		t.Fatal(err)
	}
	return func() {
		if err := ioutil.WriteFile(path, old, 0666); err != nil {
			t.Fatal(err)
		}
	}
}

// isNew returns if the path is newer than the time stamp used by touch.
func isNew(t *testing.T, path string) bool {
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return fi.ModTime().After(stampTime)
}

// Fail unless path has been rebuilt (i.e. is newer than the time stamp used by
// isNew)
func AssertRebuilt(t *testing.T, msg, path string) {
	t.Helper()
	if !isNew(t, path) {
		t.Errorf("%s was not rebuilt (%s)", msg, path)
	}
}

// Fail if path has been rebuilt (i.e. is newer than the time stamp used by isNew)
func AssertNotRebuilt(t *testing.T, msg, path string) {
	t.Helper()
	if isNew(t, path) {
		t.Errorf("%s was rebuilt (%s)", msg, path)
	}
}

func TestRebuilding(t *testing.T) {
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "depBase")
	goCmd(t, "install", "-linkshared", "exe")

	// If the source is newer than both the .a file and the .so, both are rebuilt.
	t.Run("newsource", func(t *testing.T) {
		resetFileStamps()
		cleanup := touch(t, "src/depBase/dep.go")
		defer func() {
			cleanup()
			goCmd(t, "install", "-linkshared", "exe")
		}()
		goCmd(t, "install", "-linkshared", "exe")
		AssertRebuilt(t, "new source", filepath.Join(gopathInstallDir, "depBase.a"))
		AssertRebuilt(t, "new source", filepath.Join(gopathInstallDir, "libdepBase.so"))
	})

	// If the .a file is newer than the .so, the .so is rebuilt (but not the .a)
	t.Run("newarchive", func(t *testing.T) {
		resetFileStamps()
		AssertNotRebuilt(t, "new .a file before build", filepath.Join(gopathInstallDir, "depBase.a"))
		goCmd(t, "list", "-linkshared", "-f={{.ImportPath}} {{.Stale}} {{.StaleReason}} {{.Target}}", "depBase")
		AssertNotRebuilt(t, "new .a file before build", filepath.Join(gopathInstallDir, "depBase.a"))
		cleanup := touch(t, filepath.Join(gopathInstallDir, "depBase.a"))
		defer func() {
			cleanup()
			goCmd(t, "install", "-v", "-linkshared", "exe")
		}()
		goCmd(t, "install", "-v", "-linkshared", "exe")
		AssertNotRebuilt(t, "new .a file", filepath.Join(gopathInstallDir, "depBase.a"))
		AssertRebuilt(t, "new .a file", filepath.Join(gopathInstallDir, "libdepBase.so"))
	})
}

func appendFile(t *testing.T, path, content string) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0660)
	if err != nil {
		t.Fatalf("os.OpenFile failed: %v", err)
	}
	defer func() {
		err := f.Close()
		if err != nil {
			t.Fatalf("f.Close failed: %v", err)
		}
	}()
	_, err = f.WriteString(content)
	if err != nil {
		t.Fatalf("f.WriteString failed: %v", err)
	}
}

func writeFile(t *testing.T, path, content string) {
	err := ioutil.WriteFile(path, []byte(content), 0644)
	if err != nil {
		t.Fatalf("ioutil.WriteFile failed: %v", err)
	}
}

func TestABIChecking(t *testing.T) {
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "depBase")
	goCmd(t, "install", "-linkshared", "exe")

	// If we make an ABI-breaking change to depBase and rebuild libp.so but not exe,
	// exe will abort with a complaint on startup.
	// This assumes adding an exported function breaks ABI, which is not true in
	// some senses but suffices for the narrow definition of ABI compatibility the
	// toolchain uses today.
	resetFileStamps()
	appendFile(t, "src/depBase/dep.go", "func ABIBreak() {}\n")
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "depBase")
	c := exec.Command("./bin/exe")
	output, err := c.CombinedOutput()
	if err == nil {
		t.Fatal("executing exe did not fail after ABI break")
	}
	scanner := bufio.NewScanner(bytes.NewReader(output))
	foundMsg := false
	const wantLine = "abi mismatch detected between the executable and libdepBase.so"
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
	// function) and rebuild libdepBase.so, exe still works, even if new function
	// is in a file by itself.
	resetFileStamps()
	writeFile(t, "src/depBase/dep2.go", "package depBase\nfunc noABIBreak() {}\n")
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "depBase")
	run(t, "after non-ABI breaking change", "./bin/exe")
}

// If a package 'explicit' imports a package 'implicit', building
// 'explicit' into a shared library implicitly includes implicit in
// the shared library. Building an executable that imports both
// explicit and implicit builds the code from implicit into the
// executable rather than fetching it from the shared library. The
// link still succeeds and the executable still runs though.
func TestImplicitInclusion(t *testing.T) {
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "explicit")
	goCmd(t, "install", "-linkshared", "implicitcmd")
	run(t, "running executable linked against library that contains same package as it", "./bin/implicitcmd")
}

// Tests to make sure that the type fields of empty interfaces and itab
// fields of nonempty interfaces are unique even across modules,
// so that interface equality works correctly.
func TestInterface(t *testing.T) {
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "iface_a")
	// Note: iface_i gets installed implicitly as a dependency of iface_a.
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "iface_b")
	goCmd(t, "install", "-linkshared", "iface")
	run(t, "running type/itab uniqueness tester", "./bin/iface")
}

// Access a global variable from a library.
func TestGlobal(t *testing.T) {
	goCmd(t, "install", "-buildmode=shared", "-linkshared", "globallib")
	goCmd(t, "install", "-linkshared", "global")
	run(t, "global executable", "./bin/global")
	AssertIsLinkedTo(t, "./bin/global", soname)
	AssertHasRPath(t, "./bin/global", gorootInstallDir)
}

// Run a test using -linkshared of an installed shared package.
// Issue 26400.
func TestTestInstalledShared(t *testing.T) {
	goCmd(nil, "test", "-linkshared", "-test.short", "sync/atomic")
}
