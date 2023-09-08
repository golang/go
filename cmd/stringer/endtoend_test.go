// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go command is not available on android

//go:build !android
// +build !android

package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/build"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/internal/typeparams"
)

// This file contains a test that compiles and runs each program in testdata
// after generating the string method for its type. The rule is that for testdata/x.go
// we run stringer -type X and then compile and run the program. The resulting
// binary panics if the String method for X is not correct, including for error cases.

func TestMain(m *testing.M) {
	if os.Getenv("STRINGER_TEST_IS_STRINGER") != "" {
		main()
		os.Exit(0)
	}

	// Inform subprocesses that they should run the cmd/stringer main instead of
	// running tests. It's a close approximation to building and running the real
	// command, and much less complicated and expensive to build and clean up.
	os.Setenv("STRINGER_TEST_IS_STRINGER", "1")

	flag.Parse()
	if testing.Verbose() {
		os.Setenv("GOPACKAGESDEBUG", "true")
	}

	os.Exit(m.Run())
}

func TestEndToEnd(t *testing.T) {
	stringer := stringerPath(t)
	// Read the testdata directory.
	fd, err := os.Open("testdata")
	if err != nil {
		t.Fatal(err)
	}
	defer fd.Close()
	names, err := fd.Readdirnames(-1)
	if err != nil {
		t.Fatalf("Readdirnames: %s", err)
	}
	if typeparams.Enabled {
		names = append(names, moreTests(t, "testdata/typeparams", "typeparams")...)
	}
	// Generate, compile, and run the test programs.
	for _, name := range names {
		if name == "typeparams" {
			// ignore the directory containing the tests with type params
			continue
		}
		if !strings.HasSuffix(name, ".go") {
			t.Errorf("%s is not a Go file", name)
			continue
		}
		if strings.HasPrefix(name, "tag_") || strings.HasPrefix(name, "vary_") {
			// This file is used for tag processing in TestTags or TestConstValueChange, below.
			continue
		}
		t.Run(name, func(t *testing.T) {
			if name == "cgo.go" && !build.Default.CgoEnabled {
				t.Skipf("cgo is not enabled for %s", name)
			}
			stringerCompileAndRun(t, t.TempDir(), stringer, typeName(name), name)
		})
	}
}

// a type name for stringer. use the last component of the file name with the .go
func typeName(fname string) string {
	// file names are known to be ascii and end .go
	base := path.Base(fname)
	return fmt.Sprintf("%c%s", base[0]+'A'-'a', base[1:len(base)-len(".go")])
}

func moreTests(t *testing.T, dirname, prefix string) []string {
	x, err := os.ReadDir(dirname)
	if err != nil {
		// error, but try the rest of the tests
		t.Errorf("can't read type param tess from %s: %v", dirname, err)
		return nil
	}
	names := make([]string, len(x))
	for i, f := range x {
		names[i] = prefix + "/" + f.Name()
	}
	return names
}

// TestTags verifies that the -tags flag works as advertised.
func TestTags(t *testing.T) {
	stringer := stringerPath(t)
	dir := t.TempDir()
	var (
		protectedConst = []byte("TagProtected")
		output         = filepath.Join(dir, "const_string.go")
	)
	for _, file := range []string{"tag_main.go", "tag_tag.go"} {
		err := copy(filepath.Join(dir, file), filepath.Join("testdata", file))
		if err != nil {
			t.Fatal(err)
		}
	}
	// Run stringer in the directory that contains the package files.
	// We cannot run stringer in the current directory for the following reasons:
	// - Versions of Go earlier than Go 1.11, do not support absolute directories as a pattern.
	// - When the current directory is inside a go module, the path will not be considered
	//   a valid path to a package.
	err := runInDir(t, dir, stringer, "-type", "Const", ".")
	if err != nil {
		t.Fatal(err)
	}
	result, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(result, protectedConst) {
		t.Fatal("tagged variable appears in untagged run")
	}
	err = os.Remove(output)
	if err != nil {
		t.Fatal(err)
	}
	err = runInDir(t, dir, stringer, "-type", "Const", "-tags", "tag", ".")
	if err != nil {
		t.Fatal(err)
	}
	result, err = os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(result, protectedConst) {
		t.Fatal("tagged variable does not appear in tagged run")
	}
}

// TestConstValueChange verifies that if a constant value changes and
// the stringer code is not regenerated, we'll get a compiler error.
func TestConstValueChange(t *testing.T) {
	stringer := stringerPath(t)
	dir := t.TempDir()
	source := filepath.Join(dir, "day.go")
	err := copy(source, filepath.Join("testdata", "day.go"))
	if err != nil {
		t.Fatal(err)
	}
	stringSource := filepath.Join(dir, "day_string.go")
	// Run stringer in the directory that contains the package files.
	err = runInDir(t, dir, stringer, "-type", "Day", "-output", stringSource)
	if err != nil {
		t.Fatal(err)
	}
	// Run the binary in the temporary directory as a sanity check.
	err = run(t, "go", "run", stringSource, source)
	if err != nil {
		t.Fatal(err)
	}
	// Overwrite the source file with a version that has changed constants.
	err = copy(source, filepath.Join("testdata", "vary_day.go"))
	if err != nil {
		t.Fatal(err)
	}
	// Unfortunately different compilers may give different error messages,
	// so there's no easy way to verify that the build failed specifically
	// because the constants changed rather than because the vary_day.go
	// file is invalid.
	//
	// Instead we'll just rely on manual inspection of the polluted test
	// output. An alternative might be to check that the error output
	// matches a set of possible error strings emitted by known
	// Go compilers.
	t.Logf("Note: the following messages should indicate an out-of-bounds compiler error\n")
	err = run(t, "go", "build", stringSource, source)
	if err == nil {
		t.Fatal("unexpected compiler success")
	}
}

var exe struct {
	path string
	err  error
	once sync.Once
}

func stringerPath(t *testing.T) string {
	testenv.NeedsExec(t)

	exe.once.Do(func() {
		exe.path, exe.err = os.Executable()
	})
	if exe.err != nil {
		t.Fatal(exe.err)
	}
	return exe.path
}

// stringerCompileAndRun runs stringer for the named file and compiles and
// runs the target binary in directory dir. That binary will panic if the String method is incorrect.
func stringerCompileAndRun(t *testing.T, dir, stringer, typeName, fileName string) {
	t.Logf("run: %s %s\n", fileName, typeName)
	source := filepath.Join(dir, path.Base(fileName))
	err := copy(source, filepath.Join("testdata", fileName))
	if err != nil {
		t.Fatalf("copying file to temporary directory: %s", err)
	}
	stringSource := filepath.Join(dir, typeName+"_string.go")
	// Run stringer in temporary directory.
	err = run(t, stringer, "-type", typeName, "-output", stringSource, source)
	if err != nil {
		t.Fatal(err)
	}
	// Run the binary in the temporary directory.
	err = run(t, "go", "run", stringSource, source)
	if err != nil {
		t.Fatal(err)
	}
}

// copy copies the from file to the to file.
func copy(to, from string) error {
	toFd, err := os.Create(to)
	if err != nil {
		return err
	}
	defer toFd.Close()
	fromFd, err := os.Open(from)
	if err != nil {
		return err
	}
	defer fromFd.Close()
	_, err = io.Copy(toFd, fromFd)
	return err
}

// run runs a single command and returns an error if it does not succeed.
// os/exec should have this function, to be honest.
func run(t testing.TB, name string, arg ...string) error {
	t.Helper()
	return runInDir(t, ".", name, arg...)
}

// runInDir runs a single command in directory dir and returns an error if
// it does not succeed.
func runInDir(t testing.TB, dir, name string, arg ...string) error {
	t.Helper()
	cmd := testenv.Command(t, name, arg...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GO111MODULE=auto")
	out, err := cmd.CombinedOutput()
	if len(out) > 0 {
		t.Logf("%s", out)
	}
	if err != nil {
		return fmt.Errorf("%v: %v", cmd, err)
	}
	return nil
}
