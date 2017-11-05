// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

var haveCGO bool

type test struct {
	offset, from, to string // specify the arguments
	fileSpecified    bool   // true if the offset or from args specify a specific file
	pkgs             map[string][]string
	wantErr          bool
	wantOut          string              // a substring expected to be in the output
	packages         map[string][]string // a map of the package name to the files contained within, which will be numbered by i.go where i is the index
}

// Test that renaming that would modify cgo files will produce an error and not modify the file.
func TestGeneratedFiles(t *testing.T) {
	if !haveCGO {
		t.Skipf("skipping test: no cgo")
	}

	tmp, bin, cleanup := buildGorename(t)
	defer cleanup()

	srcDir := filepath.Join(tmp, "src")
	err := os.Mkdir(srcDir, os.ModePerm)
	if err != nil {
		t.Fatal(err)
	}

	var env = []string{fmt.Sprintf("GOPATH=%s", tmp)}
	for _, envVar := range os.Environ() {
		if !strings.HasPrefix(envVar, "GOPATH=") {
			env = append(env, envVar)
		}
	}

	// Testing renaming in packages that include cgo files:
	for iter, renameTest := range []test{
		{
			// Test: variable not used in any cgo file -> no error
			from: `"mytest"::f`, to: "g",
			packages: map[string][]string{
				"mytest": []string{`package mytest; func f() {}`,
					`package mytest
// #include <stdio.h>
import "C"

func z() {C.puts(nil)}`},
			},
			wantErr: false,
			wantOut: "Renamed 1 occurrence in 1 file in 1 package.",
		}, {
			// Test: to name used in cgo file -> rename error
			from: `"mytest"::f`, to: "g",
			packages: map[string][]string{
				"mytest": []string{`package mytest; func f() {}`,
					`package mytest
// #include <stdio.h>
import "C"

func g() {C.puts(nil)}`},
			},
			wantErr: true,
			wantOut: "conflicts with func in same block",
		},
		{
			// Test: from name in package in cgo file -> error
			from: `"mytest"::f`, to: "g",
			packages: map[string][]string{
				"mytest": []string{`package mytest

// #include <stdio.h>
import "C"

func f() { C.puts(nil); }
`},
			},
			wantErr: true,
			wantOut: "gorename: refusing to modify generated file containing DO NOT EDIT marker:",
		}, {
			// Test: from name in cgo file -> error
			from: filepath.Join("mytest", "0.go") + `::f`, to: "g",
			fileSpecified: true,
			packages: map[string][]string{
				"mytest": []string{`package mytest

// #include <stdio.h>
import "C"

func f() { C.puts(nil); }
`},
			},
			wantErr: true,
			wantOut: "gorename: refusing to modify generated file containing DO NOT EDIT marker:",
		}, {
			// Test: offset in cgo file -> identifier in cgo error
			offset: filepath.Join("main", "0.go") + `:#78`, to: "bar",
			fileSpecified: true,
			wantErr:       true,
			packages: map[string][]string{
				"main": {`package main

// #include <unistd.h>
import "C"
import "fmt"

func main() {
	foo := 1
	C.close(2)
	fmt.Println(foo)
}
`},
			},
			wantOut: "cannot rename identifiers in generated file containing DO NOT EDIT marker:",
		}, {
			// Test: from identifier appears in cgo file in another package -> error
			from: `"test"::Foo`, to: "Bar",
			packages: map[string][]string{
				"test": []string{
					`package test

func Foo(x int) (int){
	return x * 2
}
`,
				},
				"main": []string{
					`package main

import "test"
import "fmt"
// #include <unistd.h>
import "C"

func fun() {
	x := test.Foo(3)
	C.close(3)
	fmt.Println(x)
}
`,
				},
			},
			wantErr: true,
			wantOut: "gorename: refusing to modify generated file containing DO NOT EDIT marker:",
		}, {
			// Test: from identifier doesn't appear in cgo file that includes modified package -> rename successful
			from: `"test".Foo::x`, to: "y",
			packages: map[string][]string{
				"test": []string{
					`package test

func Foo(x int) (int){
	return x * 2
}
`,
				},
				"main": []string{
					`package main
import "test"
import "fmt"
// #include <unistd.h>
import "C"

func fun() {
	x := test.Foo(3)
	C.close(3)
	fmt.Println(x)
}
`,
				},
			},
			wantErr: false,
			wantOut: "Renamed 2 occurrences in 1 file in 1 package.",
		}, {
			// Test: from name appears in cgo file in same package -> error
			from: `"mytest"::f`, to: "g",
			packages: map[string][]string{
				"mytest": []string{`package mytest; func f() {}`,
					`package mytest
// #include <stdio.h>
import "C"

func z() {C.puts(nil); f()}`,
					`package mytest
// #include <unistd.h>
import "C"

func foo() {C.close(3); f()}`,
				},
			},
			wantErr: true,
			wantOut: "gorename: refusing to modify generated files containing DO NOT EDIT marker:",
		}, {
			// Test: from name in file, identifier not used in cgo file -> rename successful
			from: filepath.Join("mytest", "0.go") + `::f`, to: "g",
			fileSpecified: true,
			packages: map[string][]string{
				"mytest": []string{`package mytest; func f() {}`,
					`package mytest
// #include <stdio.h>
import "C"

func z() {C.puts(nil)}`},
			},
			wantErr: false,
			wantOut: "Renamed 1 occurrence in 1 file in 1 package.",
		}, {
			// Test: from identifier imported to another package but does not modify cgo file -> rename successful
			from: `"test".Foo`, to: "Bar",
			packages: map[string][]string{
				"test": []string{
					`package test

func Foo(x int) (int){
	return x * 2
}
`,
				},
				"main": []string{
					`package main
// #include <unistd.h>
import "C"

func fun() {
	C.close(3)
}
`,
					`package main
import "test"
import "fmt"
func g() { fmt.Println(test.Foo(3)) }
`,
				},
			},
			wantErr: false,
			wantOut: "Renamed 2 occurrences in 2 files in 2 packages.",
		},
	} {
		// Write the test files
		testCleanup := setUpPackages(t, srcDir, renameTest.packages)

		// Set up arguments
		var args []string

		var arg, val string
		if renameTest.offset != "" {
			arg, val = "-offset", renameTest.offset
		} else {
			arg, val = "-from", renameTest.from
		}

		prefix := fmt.Sprintf("%d: %s %q -to %q", iter, arg, val, renameTest.to)

		if renameTest.fileSpecified {
			// add the src dir to the value of the argument
			val = filepath.Join(srcDir, val)
		}

		args = append(args, arg, val, "-to", renameTest.to)

		// Run command
		cmd := exec.Command(bin, args...)
		cmd.Args[0] = "gorename"
		cmd.Env = env

		// Check the output
		out, err := cmd.CombinedOutput()
		// errors should result in no changes to files
		if err != nil {
			if !renameTest.wantErr {
				t.Errorf("%s: received unexpected error %s", prefix, err)
			}
			// Compare output
			if ok := strings.Contains(string(out), renameTest.wantOut); !ok {
				t.Errorf("%s: unexpected command output: %s (want: %s)", prefix, out, renameTest.wantOut)
			}
			// Check that no files were modified
			if modified := modifiedFiles(t, srcDir, renameTest.packages); len(modified) != 0 {
				t.Errorf("%s: files unexpectedly modified: %s", prefix, modified)
			}

		} else {
			if !renameTest.wantErr {
				if ok := strings.Contains(string(out), renameTest.wantOut); !ok {
					t.Errorf("%s: unexpected command output: %s (want: %s)", prefix, out, renameTest.wantOut)
				}
			} else {
				t.Errorf("%s: command succeeded unexpectedly, output: %s", prefix, out)
			}
		}
		testCleanup()
	}
}

// buildGorename builds the gorename executable.
// It returns its path, and a cleanup function.
func buildGorename(t *testing.T) (tmp, bin string, cleanup func()) {

	tmp, err := ioutil.TempDir("", "gorename-regtest-")
	if err != nil {
		t.Fatal(err)
	}

	defer func() {
		if cleanup == nil { // probably, go build failed.
			os.RemoveAll(tmp)
		}
	}()

	bin = filepath.Join(tmp, "gorename")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	cmd := exec.Command("go", "build", "-o", bin)
	if err := cmd.Run(); err != nil {
		t.Fatalf("Building gorename: %v", err)
	}
	return tmp, bin, func() { os.RemoveAll(tmp) }
}

// setUpPackages sets up the files in a temporary directory provided by arguments.
func setUpPackages(t *testing.T, dir string, packages map[string][]string) (cleanup func()) {
	var pkgDirs []string

	for pkgName, files := range packages {
		// Create a directory for the package.
		pkgDir := filepath.Join(dir, pkgName)
		pkgDirs = append(pkgDirs, pkgDir)

		if err := os.Mkdir(pkgDir, os.ModePerm); err != nil {
			t.Fatal(err)
		}
		// Write the packages files
		for i, val := range files {
			file := filepath.Join(pkgDir, strconv.Itoa(i)+".go")
			if err := ioutil.WriteFile(file, []byte(val), os.ModePerm); err != nil {
				t.Fatal(err)
			}
		}
	}
	return func() {
		for _, dir := range pkgDirs {
			os.RemoveAll(dir)
		}
	}
}

// modifiedFiles returns a list of files that were renamed (without the prefix dir).
func modifiedFiles(t *testing.T, dir string, packages map[string][]string) (results []string) {

	for pkgName, files := range packages {
		pkgDir := filepath.Join(dir, pkgName)

		for i, val := range files {
			file := filepath.Join(pkgDir, strconv.Itoa(i)+".go")
			// read file contents and compare to val
			if contents, err := ioutil.ReadFile(file); err != nil {
				t.Fatalf("File missing: %s", err)
			} else if string(contents) != val {
				results = append(results, strings.TrimPrefix(dir, file))
			}
		}
	}
	return results
}
