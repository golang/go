// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// This file contains code generation tests.
//
// Each test is defined in a variable of type asmTest. Tests are
// architecture-specific, and they are grouped in arrays of tests, one
// for each architecture.
//
// Each asmTest consists of a function to compile, an array of
// positive regexps that must match the generated assembly and
// an array of negative regexps that must not match generated assembly.
// For example, the following amd64 test
//
//   {
// 	  fn: `
// 	  func f0(x int) int {
// 		  return x * 64
// 	  }
// 	  `,
// 	  pos: []string{"\tSHLQ\t[$]6,"},
//	  neg: []string{"MULQ"}
//   }
//
// verifies that the code the compiler generates for a multiplication
// by 64 contains a 'SHLQ' instruction and does not contain a MULQ.
//
// Since all the tests for a given architecture are dumped in the same
// file, the function names must be unique. As a workaround for this
// restriction, the test harness supports the use of a '$' placeholder
// for function names. The func f0 above can be also written as
//
//   {
// 	  fn: `
// 	  func $(x int) int {
// 		  return x * 64
// 	  }
// 	  `,
// 	  pos: []string{"\tSHLQ\t[$]6,"},
//	  neg: []string{"MULQ"}
//   }
//
// Each '$'-function will be given a unique name of form f<N>_<arch>,
// where <N> is the test index in the test array, and <arch> is the
// test's architecture.
//
// It is allowed to mix named and unnamed functions in the same test
// array; the named functions will retain their original names.

// TestAssembly checks to make sure the assembly generated for
// functions contains certain expected instructions.
func TestAssembly(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	if runtime.GOOS == "windows" {
		// TODO: remove if we can get "go tool compile -S" to work on windows.
		t.Skipf("skipping test: recursive windows compile not working")
	}
	dir, err := ioutil.TempDir("", "TestAssembly")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	nameRegexp := regexp.MustCompile("func \\w+")
	t.Run("platform", func(t *testing.T) {
		for _, ats := range allAsmTests {
			ats := ats
			t.Run(ats.os+"/"+ats.arch, func(tt *testing.T) {
				tt.Parallel()

				asm := ats.compileToAsm(tt, dir)

				for i, at := range ats.tests {
					var funcName string
					if strings.Contains(at.fn, "func $") {
						funcName = fmt.Sprintf("f%d_%s", i, ats.arch)
					} else {
						funcName = nameRegexp.FindString(at.fn)[len("func "):]
					}
					fa := funcAsm(tt, asm, funcName)
					if fa != "" {
						at.verifyAsm(tt, fa)
					}
				}
			})
		}
	})
}

var nextTextRegexp = regexp.MustCompile(`\n\S`)

// funcAsm returns the assembly listing for the given function name.
func funcAsm(t *testing.T, asm string, funcName string) string {
	if i := strings.Index(asm, fmt.Sprintf("TEXT\t\"\".%s(SB)", funcName)); i >= 0 {
		asm = asm[i:]
	} else {
		t.Errorf("could not find assembly for function %v", funcName)
		return ""
	}

	// Find the next line that doesn't begin with whitespace.
	loc := nextTextRegexp.FindStringIndex(asm)
	if loc != nil {
		asm = asm[:loc[0]]
	}

	return asm
}

type asmTest struct {
	// function to compile
	fn string
	// regular expressions that must match the generated assembly
	pos []string
	// regular expressions that must not match the generated assembly
	neg []string
}

func (at asmTest) verifyAsm(t *testing.T, fa string) {
	for _, r := range at.pos {
		if b, err := regexp.MatchString(r, fa); !b || err != nil {
			t.Errorf("expected:%s\ngo:%s\nasm:%s\n", r, at.fn, fa)
		}
	}
	for _, r := range at.neg {
		if b, err := regexp.MatchString(r, fa); b || err != nil {
			t.Errorf("not expected:%s\ngo:%s\nasm:%s\n", r, at.fn, fa)
		}
	}
}

type asmTests struct {
	arch    string
	os      string
	imports []string
	tests   []*asmTest
}

func (ats *asmTests) generateCode() []byte {
	var buf bytes.Buffer
	fmt.Fprintln(&buf, "package main")
	for _, s := range ats.imports {
		fmt.Fprintf(&buf, "import %q\n", s)
	}

	for i, t := range ats.tests {
		function := strings.Replace(t.fn, "func $", fmt.Sprintf("func f%d_%s", i, ats.arch), 1)
		fmt.Fprintln(&buf, function)
	}

	return buf.Bytes()
}

// compile compiles the package pkg for architecture arch and
// returns the generated assembly.  dir is a scratch directory.
func (ats *asmTests) compileToAsm(t *testing.T, dir string) string {
	// create test directory
	testDir := filepath.Join(dir, fmt.Sprintf("%s_%s", ats.arch, ats.os))
	err := os.Mkdir(testDir, 0700)
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}

	// Create source.
	src := filepath.Join(testDir, "test.go")
	err = ioutil.WriteFile(src, ats.generateCode(), 0600)
	if err != nil {
		t.Fatalf("error writing code: %v", err)
	}

	// First, install any dependencies we need.  This builds the required export data
	// for any packages that are imported.
	for _, i := range ats.imports {
		out := filepath.Join(testDir, i+".a")

		if s := ats.runGo(t, "build", "-o", out, "-gcflags=-dolinkobj=false", i); s != "" {
			t.Fatalf("Stdout = %s\nWant empty", s)
		}
	}

	// Now, compile the individual file for which we want to see the generated assembly.
	asm := ats.runGo(t, "tool", "compile", "-I", testDir, "-S", "-o", filepath.Join(testDir, "out.o"), src)
	return asm
}

// runGo runs go command with the given args and returns stdout string.
// go is run with GOARCH and GOOS set as ats.arch and ats.os respectively
func (ats *asmTests) runGo(t *testing.T, args ...string) string {
	var stdout, stderr bytes.Buffer
	cmd := exec.Command(testenv.GoToolPath(t), args...)
	cmd.Env = append(os.Environ(), "GOARCH="+ats.arch, "GOOS="+ats.os)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		t.Fatalf("error running cmd: %v\nstdout:\n%sstderr:\n%s\n", err, stdout.String(), stderr.String())
	}

	if s := stderr.String(); s != "" {
		t.Fatalf("Stderr = %s\nWant empty", s)
	}

	return stdout.String()
}

var allAsmTests = []*asmTests{
	{
		arch:  "amd64",
		os:    "plan9",
		tests: plan9AMD64Tests,
	},
}

var plan9AMD64Tests = []*asmTest{
	// We should make sure that the compiler doesn't generate floating point
	// instructions for non-float operations on Plan 9, because floating point
	// operations are not allowed in the note handler.
	// Array zeroing.
	{
		fn: `
		func $() [16]byte {
			var a [16]byte
			return a
		}
		`,
		pos: []string{"\tMOVQ\t\\$0, \"\""},
	},
	// Array copy.
	{
		fn: `
		func $(a [16]byte) (b [16]byte) {
			b = a
			return
		}
		`,
		pos: []string{"\tMOVQ\t\"\"\\.a\\+[0-9]+\\(SP\\), (AX|CX)", "\tMOVQ\t(AX|CX), \"\"\\.b\\+[0-9]+\\(SP\\)"},
	},
}

// TestLineNumber checks to make sure the generated assembly has line numbers
// see issue #16214
func TestLineNumber(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir, err := ioutil.TempDir("", "TestLineNumber")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	src := filepath.Join(dir, "x.go")
	err = ioutil.WriteFile(src, []byte(issue16214src), 0644)
	if err != nil {
		t.Fatalf("could not write file: %v", err)
	}

	cmd := exec.Command(testenv.GoToolPath(t), "tool", "compile", "-S", "-o", filepath.Join(dir, "out.o"), src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("fail to run go tool compile: %v", err)
	}

	if strings.Contains(string(out), "unknown line number") {
		t.Errorf("line number missing in assembly:\n%s", out)
	}
}

var issue16214src = `
package main

func Mod32(x uint32) uint32 {
	return x % 3 // frontend rewrites it as HMUL with 2863311531, the LITERAL node has unknown Pos
}
`
