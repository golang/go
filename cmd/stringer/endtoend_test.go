// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go command is not available on android

// +build !android

package main

import (
	"bytes"
	"fmt"
	"go/build"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// This file contains a test that compiles and runs each program in testdata
// after generating the string method for its type. The rule is that for testdata/x.go
// we run stringer -type X and then compile and run the program. The resulting
// binary panics if the String method for X is not correct, including for error cases.

func TestEndToEnd(t *testing.T) {
	dir, stringer := buildStringer(t)
	defer os.RemoveAll(dir)
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
	// Generate, compile, and run the test programs.
	for _, name := range names {
		if !strings.HasSuffix(name, ".go") {
			t.Errorf("%s is not a Go file", name)
			continue
		}
		if strings.HasPrefix(name, "tag_") {
			// This file is used for tag processing in TestTags, below.
			continue
		}
		if name == "cgo.go" && !build.Default.CgoEnabled {
			t.Logf("cgo is not enabled for %s", name)
			continue
		}
		// Names are known to be ASCII and long enough.
		typeName := fmt.Sprintf("%c%s", name[0]+'A'-'a', name[1:len(name)-len(".go")])
		stringerCompileAndRun(t, dir, stringer, typeName, name)
	}
}

// TestTags verifies that the -tags flag works as advertised.
func TestTags(t *testing.T) {
	dir, stringer := buildStringer(t)
	defer os.RemoveAll(dir)
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
	err := runInDir(dir, stringer, "-type", "Const", ".")
	if err != nil {
		t.Fatal(err)
	}
	result, err := ioutil.ReadFile(output)
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
	err = runInDir(dir, stringer, "-type", "Const", "-tags", "tag", ".")
	if err != nil {
		t.Fatal(err)
	}
	result, err = ioutil.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(result, protectedConst) {
		t.Fatal("tagged variable does not appear in tagged run")
	}
}

// buildStringer creates a temporary directory and installs stringer there.
func buildStringer(t *testing.T) (dir string, stringer string) {
	t.Helper()
	dir, err := ioutil.TempDir("", "stringer")
	if err != nil {
		t.Fatal(err)
	}
	stringer = filepath.Join(dir, "stringer.exe")
	err = run("go", "build", "-o", stringer)
	if err != nil {
		t.Fatalf("building stringer: %s", err)
	}
	return dir, stringer
}

// stringerCompileAndRun runs stringer for the named file and compiles and
// runs the target binary in directory dir. That binary will panic if the String method is incorrect.
func stringerCompileAndRun(t *testing.T, dir, stringer, typeName, fileName string) {
	t.Helper()
	t.Logf("run: %s %s\n", fileName, typeName)
	source := filepath.Join(dir, fileName)
	err := copy(source, filepath.Join("testdata", fileName))
	if err != nil {
		t.Fatalf("copying file to temporary directory: %s", err)
	}
	stringSource := filepath.Join(dir, typeName+"_string.go")
	// Run stringer in temporary directory.
	err = run(stringer, "-type", typeName, "-output", stringSource, source)
	if err != nil {
		t.Fatal(err)
	}
	// Run the binary in the temporary directory.
	err = run("go", "run", stringSource, source)
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
func run(name string, arg ...string) error {
	return runInDir(".", name, arg...)
}

// runInDir runs a single command in directory dir and returns an error if
// it does not succeed.
func runInDir(dir, name string, arg ...string) error {
	cmd := exec.Command(name, arg...)
	cmd.Dir = dir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}
