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
		arch:    "amd64",
		os:      "linux",
		imports: []string{"runtime"},
		tests:   linuxAMD64Tests,
	},
	{
		arch:    "arm",
		os:      "linux",
		imports: []string{"runtime"},
		tests:   linuxARMTests,
	},
	{
		arch:  "arm64",
		os:    "linux",
		tests: linuxARM64Tests,
	},
	{
		arch:  "mips64",
		os:    "linux",
		tests: linuxMIPS64Tests,
	},
	{
		arch:  "amd64",
		os:    "plan9",
		tests: plan9AMD64Tests,
	},
}

var linuxAMD64Tests = []*asmTest{
	{
		fn: `
		func $(x int) int {
			return x * 96
		}
		`,
		pos: []string{"\tSHLQ\t\\$5,", "\tLEAQ\t\\(.*\\)\\(.*\\*2\\),"},
	},
	// see issue 19595.
	// We want to merge load+op in f58, but not in f59.
	{
		fn: `
		func f58(p, q *int) {
			x := *p
			*q += x
		}`,
		pos: []string{"\tADDQ\t\\("},
	},
	{
		fn: `
		func f59(p, q *int) {
			x := *p
			for i := 0; i < 10; i++ {
				*q += x
			}
		}`,
		pos: []string{"\tADDQ\t[A-Z]"},
	},
	{
		// make sure assembly output has matching offset and base register.
		fn: `
		func f72(a, b int) int {
			runtime.GC() // use some frame
			return b
		}
		`,
		pos: []string{"b\\+24\\(SP\\)"},
	},
	{
		// check load combining
		fn: `
		func f73(a, b byte) (byte,byte) {
		    return f73(f73(a,b))
		}
		`,
		pos: []string{"\tMOVW\t"},
	},
	{
		fn: `
		func f74(a, b uint16) (uint16,uint16) {
		    return f74(f74(a,b))
		}
		`,
		pos: []string{"\tMOVL\t"},
	},
	{
		fn: `
		func f75(a, b uint32) (uint32,uint32) {
		    return f75(f75(a,b))
		}
		`,
		pos: []string{"\tMOVQ\t"},
	},
	// Make sure we don't put pointers in SSE registers across safe points.
	{
		fn: `
		func $(p, q *[2]*int)  {
		    a, b := p[0], p[1]
		    runtime.GC()
		    q[0], q[1] = a, b
		}
		`,
		neg: []string{"MOVUPS"},
	},
}

var linuxARMTests = []*asmTest{
	{
		// make sure assembly output has matching offset and base register.
		fn: `
		func f13(a, b int) int {
			runtime.GC() // use some frame
			return b
		}
		`,
		pos: []string{"b\\+4\\(FP\\)"},
	},
}

var linuxARM64Tests = []*asmTest{
	{
		fn: `
		func $(x, y uint32) uint32 {
			return x &^ y
		}
		`,
		pos: []string{"\tBIC\t"},
		neg: []string{"\tAND\t"},
	},
	{
		fn: `
		func $(x, y uint32) uint32 {
			return x ^ ^y
		}
		`,
		pos: []string{"\tEON\t"},
		neg: []string{"\tXOR\t"},
	},
	{
		fn: `
		func $(x, y uint32) uint32 {
			return x | ^y
		}
		`,
		pos: []string{"\tORN\t"},
		neg: []string{"\tORR\t"},
	},
	{
		fn: `
		func f34(a uint64) uint64 {
			return a & ((1<<63)-1)
		}
		`,
		pos: []string{"\tAND\t"},
	},
	{
		fn: `
		func f35(a uint64) uint64 {
			return a & (1<<63)
		}
		`,
		pos: []string{"\tAND\t"},
	},
	{
		// make sure offsets are folded into load and store.
		fn: `
		func f36(_, a [20]byte) (b [20]byte) {
			b = a
			return
		}
		`,
		pos: []string{"\tMOVD\t\"\"\\.a\\+[0-9]+\\(FP\\), R[0-9]+", "\tMOVD\tR[0-9]+, \"\"\\.b\\+[0-9]+\\(FP\\)"},
	},
	{
		// check that we don't emit comparisons for constant shift
		fn: `
//go:nosplit
		func $(x int) int {
			return x << 17
		}
		`,
		pos: []string{"LSL\t\\$17"},
		neg: []string{"CMP"},
	},
	// Load-combining tests.
	{
		fn: `
		func $(s []byte) uint16 {
			return uint16(s[0]) | uint16(s[1]) << 8
		}
		`,
		pos: []string{"\tMOVHU\t\\(R[0-9]+\\)"},
		neg: []string{"ORR\tR[0-9]+<<8\t"},
	},
	{
		// make sure that CSEL is emitted for conditional moves
		fn: `
		func f37(c int) int {
		     x := c + 4
		     if c < 0 {
		     	x = 182
		     }
		     return x
		}
		`,
		pos: []string{"\tCSEL\t"},
	},
}

var linuxMIPS64Tests = []*asmTest{
	{
		// check that we don't emit comparisons for constant shift
		fn: `
		func $(x int) int {
			return x << 17
		}
		`,
		pos: []string{"SLLV\t\\$17"},
		neg: []string{"SGT"},
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
