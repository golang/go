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
		imports: []string{"encoding/binary", "math", "math/bits", "unsafe", "runtime"},
		tests:   linuxAMD64Tests,
	},
	{
		arch:    "386",
		os:      "linux",
		imports: []string{"encoding/binary"},
		tests:   linux386Tests,
	},
	{
		arch:    "s390x",
		os:      "linux",
		imports: []string{"encoding/binary", "math", "math/bits"},
		tests:   linuxS390XTests,
	},
	{
		arch:    "arm",
		os:      "linux",
		imports: []string{"math/bits", "runtime"},
		tests:   linuxARMTests,
	},
	{
		arch:    "arm64",
		os:      "linux",
		imports: []string{"math/bits"},
		tests:   linuxARM64Tests,
	},
	{
		arch:    "mips",
		os:      "linux",
		imports: []string{"math/bits"},
		tests:   linuxMIPSTests,
	},
	{
		arch:  "mips64",
		os:    "linux",
		tests: linuxMIPS64Tests,
	},
	{
		arch:    "ppc64le",
		os:      "linux",
		imports: []string{"encoding/binary", "math", "math/bits"},
		tests:   linuxPPC64LETests,
	},
	{
		arch:  "amd64",
		os:    "plan9",
		tests: plan9AMD64Tests,
	},
}

var linuxAMD64Tests = []*asmTest{
	// multiplication by powers of two
	{
		fn: `
		func $(n int) int {
			return n * 64
		}
		`,
		pos: []string{"\tSHLQ\t\\$6,"},
		neg: []string{"IMULQ"},
	},
	{
		fn: `
		func $(n int) int {
			return -128*n
		}
		`,
		pos: []string{"SHLQ"},
		neg: []string{"IMULQ"},
	},

	{
		fn: `
		func $(x int) int {
			return x * 96
		}
		`,
		pos: []string{"\tSHLQ\t\\$5,", "\tLEAQ\t\\(.*\\)\\(.*\\*2\\),"},
	},
	// Load-combining tests.
	{
		fn: `
		func f2(b []byte) uint64 {
			return binary.LittleEndian.Uint64(b)
		}
		`,
		pos: []string{"\tMOVQ\t\\(.*\\),"},
	},
	{
		fn: `
		func f3(b []byte, i int) uint64 {
			return binary.LittleEndian.Uint64(b[i:])
		}
		`,
		pos: []string{"\tMOVQ\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{
		fn: `
		func f4(b []byte) uint32 {
			return binary.LittleEndian.Uint32(b)
		}
		`,
		pos: []string{"\tMOVL\t\\(.*\\),"},
	},
	{
		fn: `
		func f5(b []byte, i int) uint32 {
			return binary.LittleEndian.Uint32(b[i:])
		}
		`,
		pos: []string{"\tMOVL\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{
		fn: `
		func f6(b []byte) uint64 {
			return binary.BigEndian.Uint64(b)
		}
		`,
		pos: []string{"\tBSWAPQ\t"},
	},
	{
		fn: `
		func f7(b []byte, i int) uint64 {
			return binary.BigEndian.Uint64(b[i:])
		}
		`,
		pos: []string{"\tBSWAPQ\t"},
	},
	{
		fn: `
		func f8(b []byte, v uint64) {
			binary.BigEndian.PutUint64(b, v)
		}
		`,
		pos: []string{"\tBSWAPQ\t"},
	},
	{
		fn: `
		func f9(b []byte, i int, v uint64) {
			binary.BigEndian.PutUint64(b[i:], v)
		}
		`,
		pos: []string{"\tBSWAPQ\t"},
	},
	{
		fn: `
		func f10(b []byte) uint32 {
			return binary.BigEndian.Uint32(b)
		}
		`,
		pos: []string{"\tBSWAPL\t"},
	},
	{
		fn: `
		func f11(b []byte, i int) uint32 {
			return binary.BigEndian.Uint32(b[i:])
		}
		`,
		pos: []string{"\tBSWAPL\t"},
	},
	{
		fn: `
		func f12(b []byte, v uint32) {
			binary.BigEndian.PutUint32(b, v)
		}
		`,
		pos: []string{"\tBSWAPL\t"},
	},
	{
		fn: `
		func f13(b []byte, i int, v uint32) {
			binary.BigEndian.PutUint32(b[i:], v)
		}
		`,
		pos: []string{"\tBSWAPL\t"},
	},
	{
		fn: `
		func f14(b []byte) uint16 {
			return binary.BigEndian.Uint16(b)
		}
		`,
		pos: []string{"\tROLW\t\\$8,"},
	},
	{
		fn: `
		func f15(b []byte, i int) uint16 {
			return binary.BigEndian.Uint16(b[i:])
		}
		`,
		pos: []string{"\tROLW\t\\$8,"},
	},
	{
		fn: `
		func f16(b []byte, v uint16) {
			binary.BigEndian.PutUint16(b, v)
		}
		`,
		pos: []string{"\tROLW\t\\$8,"},
	},
	{
		fn: `
		func f17(b []byte, i int, v uint16) {
			binary.BigEndian.PutUint16(b[i:], v)
		}
		`,
		pos: []string{"\tROLW\t\\$8,"},
	},
	// Structure zeroing.  See issue #18370.
	{
		fn: `
		type T1 struct {
			a, b, c int
		}
		func $(t *T1) {
			*t = T1{}
		}
		`,
		pos: []string{"\tXORPS\tX., X", "\tMOVUPS\tX., \\(.*\\)", "\tMOVQ\t\\$0, 16\\(.*\\)"},
	},
	// SSA-able composite literal initialization. Issue 18872.
	{
		fn: `
		type T18872 struct {
			a, b, c, d int
		}

		func f18872(p *T18872) {
			*p = T18872{1, 2, 3, 4}
		}
		`,
		pos: []string{"\tMOVQ\t[$]1", "\tMOVQ\t[$]2", "\tMOVQ\t[$]3", "\tMOVQ\t[$]4"},
	},
	// Also test struct containing pointers (this was special because of write barriers).
	{
		fn: `
		type T2 struct {
			a, b, c *int
		}
		func f19(t *T2) {
			*t = T2{}
		}
		`,
		pos: []string{"\tXORPS\tX., X", "\tMOVUPS\tX., \\(.*\\)", "\tMOVQ\t\\$0, 16\\(.*\\)", "\tCALL\truntime\\.(writebarrierptr|gcWriteBarrier)\\(SB\\)"},
	},
	// Rotate tests
	{
		fn: `
		func f20(x uint64) uint64 {
			return x<<7 | x>>57
		}
		`,
		pos: []string{"\tROLQ\t[$]7,"},
	},
	{
		fn: `
		func f21(x uint64) uint64 {
			return x<<7 + x>>57
		}
		`,
		pos: []string{"\tROLQ\t[$]7,"},
	},
	{
		fn: `
		func f22(x uint64) uint64 {
			return x<<7 ^ x>>57
		}
		`,
		pos: []string{"\tROLQ\t[$]7,"},
	},
	{
		fn: `
		func f23(x uint32) uint32 {
			return x<<7 + x>>25
		}
		`,
		pos: []string{"\tROLL\t[$]7,"},
	},
	{
		fn: `
		func f24(x uint32) uint32 {
			return x<<7 | x>>25
		}
		`,
		pos: []string{"\tROLL\t[$]7,"},
	},
	{
		fn: `
		func f25(x uint32) uint32 {
			return x<<7 ^ x>>25
		}
		`,
		pos: []string{"\tROLL\t[$]7,"},
	},
	{
		fn: `
		func f26(x uint16) uint16 {
			return x<<7 + x>>9
		}
		`,
		pos: []string{"\tROLW\t[$]7,"},
	},
	{
		fn: `
		func f27(x uint16) uint16 {
			return x<<7 | x>>9
		}
		`,
		pos: []string{"\tROLW\t[$]7,"},
	},
	{
		fn: `
		func f28(x uint16) uint16 {
			return x<<7 ^ x>>9
		}
		`,
		pos: []string{"\tROLW\t[$]7,"},
	},
	{
		fn: `
		func f29(x uint8) uint8 {
			return x<<7 + x>>1
		}
		`,
		pos: []string{"\tROLB\t[$]7,"},
	},
	{
		fn: `
		func f30(x uint8) uint8 {
			return x<<7 | x>>1
		}
		`,
		pos: []string{"\tROLB\t[$]7,"},
	},
	{
		fn: `
		func f31(x uint8) uint8 {
			return x<<7 ^ x>>1
		}
		`,
		pos: []string{"\tROLB\t[$]7,"},
	},
	// Rotate after inlining (see issue 18254).
	{
		fn: `
		func f32(x uint32) uint32 {
			return g(x, 7)
		}
		func g(x uint32, k uint) uint32 {
			return x<<k | x>>(32-k)
		}
		`,
		pos: []string{"\tROLL\t[$]7,"},
	},
	{
		fn: `
		func f33(m map[int]int) int {
			return m[5]
		}
		`,
		pos: []string{"\tMOVQ\t[$]5,"},
	},
	// Direct use of constants in fast map access calls. Issue 19015.
	{
		fn: `
		func f34(m map[int]int) bool {
			_, ok := m[5]
			return ok
		}
		`,
		pos: []string{"\tMOVQ\t[$]5,"},
	},
	{
		fn: `
		func f35(m map[string]int) int {
			return m["abc"]
		}
		`,
		pos: []string{"\"abc\""},
	},
	{
		fn: `
		func f36(m map[string]int) bool {
			_, ok := m["abc"]
			return ok
		}
		`,
		pos: []string{"\"abc\""},
	},
	// Bit test ops on amd64, issue 18943.
	{
		fn: `
		func f37(a, b uint64) int {
			if a&(1<<(b&63)) != 0 {
				return 1
			}
			return -1
		}
		`,
		pos: []string{"\tBTQ\t"},
	},
	{
		fn: `
		func f38(a, b uint64) bool {
			return a&(1<<(b&63)) != 0
		}
		`,
		pos: []string{"\tBTQ\t"},
	},
	{
		fn: `
		func f39(a uint64) int {
			if a&(1<<60) != 0 {
				return 1
			}
			return -1
		}
		`,
		pos: []string{"\tBTQ\t\\$60"},
	},
	{
		fn: `
		func f40(a uint64) bool {
			return a&(1<<60) != 0
		}
		`,
		pos: []string{"\tBTQ\t\\$60"},
	},
	// Intrinsic tests for math/bits
	{
		fn: `
		func f41(a uint64) int {
			return bits.TrailingZeros64(a)
		}
		`,
		pos: []string{"\tBSFQ\t", "\tMOVL\t\\$64,", "\tCMOVQEQ\t"},
	},
	{
		fn: `
		func f42(a uint32) int {
			return bits.TrailingZeros32(a)
		}
		`,
		pos: []string{"\tBSFQ\t", "\tORQ\t[^$]", "\tMOVQ\t\\$4294967296,"},
	},
	{
		fn: `
		func f43(a uint16) int {
			return bits.TrailingZeros16(a)
		}
		`,
		pos: []string{"\tBSFQ\t", "\tORQ\t\\$65536,"},
	},
	{
		fn: `
		func f44(a uint8) int {
			return bits.TrailingZeros8(a)
		}
		`,
		pos: []string{"\tBSFQ\t", "\tORQ\t\\$256,"},
	},
	{
		fn: `
		func f45(a uint64) uint64 {
			return bits.ReverseBytes64(a)
		}
		`,
		pos: []string{"\tBSWAPQ\t"},
	},
	{
		fn: `
		func f46(a uint32) uint32 {
			return bits.ReverseBytes32(a)
		}
		`,
		pos: []string{"\tBSWAPL\t"},
	},
	{
		fn: `
		func f47(a uint16) uint16 {
			return bits.ReverseBytes16(a)
		}
		`,
		pos: []string{"\tROLW\t\\$8,"},
	},
	{
		fn: `
		func f48(a uint64) int {
			return bits.Len64(a)
		}
		`,
		pos: []string{"\tBSRQ\t"},
	},
	{
		fn: `
		func f49(a uint32) int {
			return bits.Len32(a)
		}
		`,
		pos: []string{"\tBSRQ\t"},
	},
	{
		fn: `
		func f50(a uint16) int {
			return bits.Len16(a)
		}
		`,
		pos: []string{"\tBSRQ\t"},
	},
	/* see ssa.go
	{
		fn:`
		func f51(a uint8) int {
			return bits.Len8(a)
		}
		`,
		pos:[]string{"\tBSRQ\t"},
	},
	*/
	{
		fn: `
		func f52(a uint) int {
			return bits.Len(a)
		}
		`,
		pos: []string{"\tBSRQ\t"},
	},
	{
		fn: `
		func f53(a uint64) int {
			return bits.LeadingZeros64(a)
		}
		`,
		pos: []string{"\tBSRQ\t"},
	},
	{
		fn: `
		func f54(a uint32) int {
			return bits.LeadingZeros32(a)
		}
		`,
		pos: []string{"\tBSRQ\t"},
	},
	{
		fn: `
		func f55(a uint16) int {
			return bits.LeadingZeros16(a)
		}
		`,
		pos: []string{"\tBSRQ\t"},
	},
	/* see ssa.go
	{
		fn:`
		func f56(a uint8) int {
			return bits.LeadingZeros8(a)
		}
		`,
		pos:[]string{"\tBSRQ\t"},
	},
	*/
	{
		fn: `
		func f57(a uint) int {
			return bits.LeadingZeros(a)
		}
		`,
		pos: []string{"\tBSRQ\t"},
	},
	{
		fn: `
		func pop1(x uint64) int {
			return bits.OnesCount64(x)
		}`,
		pos: []string{"\tPOPCNTQ\t", "support_popcnt"},
	},
	{
		fn: `
		func pop2(x uint32) int {
			return bits.OnesCount32(x)
		}`,
		pos: []string{"\tPOPCNTL\t", "support_popcnt"},
	},
	{
		fn: `
		func pop3(x uint16) int {
			return bits.OnesCount16(x)
		}`,
		pos: []string{"\tPOPCNTL\t", "support_popcnt"},
	},
	{
		fn: `
		func pop4(x uint) int {
			return bits.OnesCount(x)
		}`,
		pos: []string{"\tPOPCNTQ\t", "support_popcnt"},
	},
	// multiplication merging tests
	{
		fn: `
		func mul1(n int) int {
			return 15*n + 31*n
		}`,
		pos: []string{"\tIMULQ\t[$]46"}, // 46*n
	},
	{
		fn: `
		func mul2(n int) int {
			return 5*n + 7*(n+1) + 11*(n+2)
		}`,
		pos: []string{"\tIMULQ\t[$]23", "\tADDQ\t[$]29"}, // 23*n + 29
	},
	{
		fn: `
		func mul3(a, n int) int {
			return a*n + 19*n
		}`,
		pos: []string{"\tADDQ\t[$]19", "\tIMULQ"}, // (a+19)*n
	},
	{
		fn: `
		func mul4(n int) int {
			return 23*n - 9*n
		}`,
		pos: []string{"\tIMULQ\t[$]14"}, // 14*n
	},
	{
		fn: `
		func mul5(a, n int) int {
			return a*n - 19*n
		}`,
		pos: []string{"\tADDQ\t[$]-19", "\tIMULQ"}, // (a-19)*n
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
	// Floating-point strength reduction
	{
		fn: `
		func f60(f float64) float64 {
			return f * 2.0
		}`,
		pos: []string{"\tADDSD\t"},
	},
	{
		fn: `
		func f62(f float64) float64 {
			return f / 16.0
		}`,
		pos: []string{"\tMULSD\t"},
	},
	{
		fn: `
		func f63(f float64) float64 {
			return f / 0.125
		}`,
		pos: []string{"\tMULSD\t"},
	},
	{
		fn: `
		func f64(f float64) float64 {
			return f / 0.5
		}`,
		pos: []string{"\tADDSD\t"},
	},
	// Check that compare to constant string uses 2/4/8 byte compares
	{
		fn: `
		func f65(a string) bool {
		    return a == "xx"
		}`,
		pos: []string{"\tCMPW\t[A-Z]"},
	},
	{
		fn: `
		func f66(a string) bool {
		    return a == "xxxx"
		}`,
		pos: []string{"\tCMPL\t[A-Z]"},
	},
	{
		fn: `
		func f67(a string) bool {
		    return a == "xxxxxxxx"
		}`,
		pos: []string{"\tCMPQ\t[A-Z]"},
	},
	// Non-constant rotate
	{
		fn: `func rot64l(x uint64, y int) uint64 {
			z := uint(y & 63)
			return x << z | x >> (64-z)
		}`,
		pos: []string{"\tROLQ\t"},
	},
	{
		fn: `func rot64r(x uint64, y int) uint64 {
			z := uint(y & 63)
			return x >> z | x << (64-z)
		}`,
		pos: []string{"\tRORQ\t"},
	},
	{
		fn: `func rot32l(x uint32, y int) uint32 {
			z := uint(y & 31)
			return x << z | x >> (32-z)
		}`,
		pos: []string{"\tROLL\t"},
	},
	{
		fn: `func rot32r(x uint32, y int) uint32 {
			z := uint(y & 31)
			return x >> z | x << (32-z)
		}`,
		pos: []string{"\tRORL\t"},
	},
	{
		fn: `func rot16l(x uint16, y int) uint16 {
			z := uint(y & 15)
			return x << z | x >> (16-z)
		}`,
		pos: []string{"\tROLW\t"},
	},
	{
		fn: `func rot16r(x uint16, y int) uint16 {
			z := uint(y & 15)
			return x >> z | x << (16-z)
		}`,
		pos: []string{"\tRORW\t"},
	},
	{
		fn: `func rot8l(x uint8, y int) uint8 {
			z := uint(y & 7)
			return x << z | x >> (8-z)
		}`,
		pos: []string{"\tROLB\t"},
	},
	{
		fn: `func rot8r(x uint8, y int) uint8 {
			z := uint(y & 7)
			return x >> z | x << (8-z)
		}`,
		pos: []string{"\tRORB\t"},
	},
	// Check that array compare uses 2/4/8 byte compares
	{
		fn: `
		func f68(a,b [2]byte) bool {
		    return a == b
		}`,
		pos: []string{"\tCMPW\t[A-Z]"},
	},
	{
		fn: `
		func f69(a,b [3]uint16) bool {
		    return a == b
		}`,
		pos: []string{"\tCMPL\t[A-Z]"},
	},
	{
		fn: `
		func $(a,b [3]int16) bool {
		    return a == b
		}`,
		pos: []string{"\tCMPL\t[A-Z]"},
	},
	{
		fn: `
		func $(a,b [12]int8) bool {
		    return a == b
		}`,
		pos: []string{"\tCMPQ\t[A-Z]", "\tCMPL\t[A-Z]"},
	},
	{
		fn: `
		func f70(a,b [15]byte) bool {
		    return a == b
		}`,
		pos: []string{"\tCMPQ\t[A-Z]"},
	},
	{
		fn: `
		func f71(a,b unsafe.Pointer) bool { // This was a TODO in mapaccess1_faststr
		    return *((*[4]byte)(a)) != *((*[4]byte)(b))
		}`,
		pos: []string{"\tCMPL\t[A-Z]"},
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
	{
		// check that stack store is optimized away
		fn: `
		func $() int {
			var x int
			return *(&x)
		}
		`,
		pos: []string{"TEXT\t.*, [$]0-8"},
	},
	// math.Abs using integer registers
	{
		fn: `
		func $(x float64) float64 {
			return math.Abs(x)
		}
		`,
		pos: []string{"\tSHLQ\t[$]1,", "\tSHRQ\t[$]1,"},
	},
	// math.Copysign using integer registers
	{
		fn: `
		func $(x, y float64) float64 {
			return math.Copysign(x, y)
		}
		`,
		pos: []string{"\tSHLQ\t[$]1,", "\tSHRQ\t[$]1,", "\tSHRQ\t[$]63,", "\tSHLQ\t[$]63,", "\tORQ\t"},
	},
	// int <-> fp moves
	{
		fn: `
		func $(x float64) uint64 {
			return math.Float64bits(x+1) + 1
		}
		`,
		pos: []string{"\tMOVQ\tX.*, [^X].*"},
	},
	{
		fn: `
		func $(x float32) uint32 {
			return math.Float32bits(x+1) + 1
		}
		`,
		pos: []string{"\tMOVL\tX.*, [^X].*"},
	},
	{
		fn: `
		func $(x uint64) float64 {
			return math.Float64frombits(x+1) + 1
		}
		`,
		pos: []string{"\tMOVQ\t[^X].*, X.*"},
	},
	{
		fn: `
		func $(x uint32) float32 {
			return math.Float32frombits(x+1) + 1
		}
		`,
		pos: []string{"\tMOVL\t[^X].*, X.*"},
	},
	{
		fn: `
		func $(x uint32) bool {
			return x > 4
		}
		`,
		pos: []string{"\tSETHI\t\\("},
	},
	// Check that len() and cap() div by a constant power of two
	// are compiled into SHRQ.
	{
		fn: `
		func $(a []int) int {
			return len(a) / 1024
		}
		`,
		pos: []string{"\tSHRQ\t\\$10,"},
	},
	{
		fn: `
		func $(s string) int {
			return len(s) / (4097 >> 1)
		}
		`,
		pos: []string{"\tSHRQ\t\\$11,"},
	},
	{
		fn: `
		func $(a []int) int {
			return cap(a) / ((1 << 11) + 2048)
		}
		`,
		pos: []string{"\tSHRQ\t\\$12,"},
	},
	// Check that len() and cap() mod by a constant power of two
	// are compiled into ANDQ.
	{
		fn: `
		func $(a []int) int {
			return len(a) % 1024
		}
		`,
		pos: []string{"\tANDQ\t\\$1023,"},
	},
	{
		fn: `
		func $(s string) int {
			return len(s) % (4097 >> 1)
		}
		`,
		pos: []string{"\tANDQ\t\\$2047,"},
	},
	{
		fn: `
		func $(a []int) int {
			return cap(a) % ((1 << 11) + 2048)
		}
		`,
		pos: []string{"\tANDQ\t\\$4095,"},
	},
	{
		// Test that small memmove was replaced with direct movs
		fn: `
                func $() {
                       x := [...]byte{1, 2, 3, 4, 5, 6, 7}
                       copy(x[1:], x[:])
                }
		`,
		neg: []string{"memmove"},
	},
	{
		// Same as above but with different size
		fn: `
                func $() {
                       x := [...]byte{1, 2, 3, 4}
                       copy(x[1:], x[:])
                }
		`,
		neg: []string{"memmove"},
	},
	{
		// Same as above but with different size
		fn: `
                func $() {
                       x := [...]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
                       copy(x[1:], x[:])
                }
		`,
		neg: []string{"memmove"},
	},
	// Nil checks before calling interface methods
	{
		fn: `
		type I interface {
			foo000()
			foo001()
			foo002()
			foo003()
			foo004()
			foo005()
			foo006()
			foo007()
			foo008()
			foo009()
			foo010()
			foo011()
			foo012()
			foo013()
			foo014()
			foo015()
			foo016()
			foo017()
			foo018()
			foo019()
			foo020()
			foo021()
			foo022()
			foo023()
			foo024()
			foo025()
			foo026()
			foo027()
			foo028()
			foo029()
			foo030()
			foo031()
			foo032()
			foo033()
			foo034()
			foo035()
			foo036()
			foo037()
			foo038()
			foo039()
			foo040()
			foo041()
			foo042()
			foo043()
			foo044()
			foo045()
			foo046()
			foo047()
			foo048()
			foo049()
			foo050()
			foo051()
			foo052()
			foo053()
			foo054()
			foo055()
			foo056()
			foo057()
			foo058()
			foo059()
			foo060()
			foo061()
			foo062()
			foo063()
			foo064()
			foo065()
			foo066()
			foo067()
			foo068()
			foo069()
			foo070()
			foo071()
			foo072()
			foo073()
			foo074()
			foo075()
			foo076()
			foo077()
			foo078()
			foo079()
			foo080()
			foo081()
			foo082()
			foo083()
			foo084()
			foo085()
			foo086()
			foo087()
			foo088()
			foo089()
			foo090()
			foo091()
			foo092()
			foo093()
			foo094()
			foo095()
			foo096()
			foo097()
			foo098()
			foo099()
			foo100()
			foo101()
			foo102()
			foo103()
			foo104()
			foo105()
			foo106()
			foo107()
			foo108()
			foo109()
			foo110()
			foo111()
			foo112()
			foo113()
			foo114()
			foo115()
			foo116()
			foo117()
			foo118()
			foo119()
			foo120()
			foo121()
			foo122()
			foo123()
			foo124()
			foo125()
			foo126()
			foo127()
			foo128()
			foo129()
			foo130()
			foo131()
			foo132()
			foo133()
			foo134()
			foo135()
			foo136()
			foo137()
			foo138()
			foo139()
			foo140()
			foo141()
			foo142()
			foo143()
			foo144()
			foo145()
			foo146()
			foo147()
			foo148()
			foo149()
			foo150()
			foo151()
			foo152()
			foo153()
			foo154()
			foo155()
			foo156()
			foo157()
			foo158()
			foo159()
			foo160()
			foo161()
			foo162()
			foo163()
			foo164()
			foo165()
			foo166()
			foo167()
			foo168()
			foo169()
			foo170()
			foo171()
			foo172()
			foo173()
			foo174()
			foo175()
			foo176()
			foo177()
			foo178()
			foo179()
			foo180()
			foo181()
			foo182()
			foo183()
			foo184()
			foo185()
			foo186()
			foo187()
			foo188()
			foo189()
			foo190()
			foo191()
			foo192()
			foo193()
			foo194()
			foo195()
			foo196()
			foo197()
			foo198()
			foo199()
			foo200()
			foo201()
			foo202()
			foo203()
			foo204()
			foo205()
			foo206()
			foo207()
			foo208()
			foo209()
			foo210()
			foo211()
			foo212()
			foo213()
			foo214()
			foo215()
			foo216()
			foo217()
			foo218()
			foo219()
			foo220()
			foo221()
			foo222()
			foo223()
			foo224()
			foo225()
			foo226()
			foo227()
			foo228()
			foo229()
			foo230()
			foo231()
			foo232()
			foo233()
			foo234()
			foo235()
			foo236()
			foo237()
			foo238()
			foo239()
			foo240()
			foo241()
			foo242()
			foo243()
			foo244()
			foo245()
			foo246()
			foo247()
			foo248()
			foo249()
			foo250()
			foo251()
			foo252()
			foo253()
			foo254()
			foo255()
			foo256()
			foo257()
			foo258()
			foo259()
			foo260()
			foo261()
			foo262()
			foo263()
			foo264()
			foo265()
			foo266()
			foo267()
			foo268()
			foo269()
			foo270()
			foo271()
			foo272()
			foo273()
			foo274()
			foo275()
			foo276()
			foo277()
			foo278()
			foo279()
			foo280()
			foo281()
			foo282()
			foo283()
			foo284()
			foo285()
			foo286()
			foo287()
			foo288()
			foo289()
			foo290()
			foo291()
			foo292()
			foo293()
			foo294()
			foo295()
			foo296()
			foo297()
			foo298()
			foo299()
			foo300()
			foo301()
			foo302()
			foo303()
			foo304()
			foo305()
			foo306()
			foo307()
			foo308()
			foo309()
			foo310()
			foo311()
			foo312()
			foo313()
			foo314()
			foo315()
			foo316()
			foo317()
			foo318()
			foo319()
			foo320()
			foo321()
			foo322()
			foo323()
			foo324()
			foo325()
			foo326()
			foo327()
			foo328()
			foo329()
			foo330()
			foo331()
			foo332()
			foo333()
			foo334()
			foo335()
			foo336()
			foo337()
			foo338()
			foo339()
			foo340()
			foo341()
			foo342()
			foo343()
			foo344()
			foo345()
			foo346()
			foo347()
			foo348()
			foo349()
			foo350()
			foo351()
			foo352()
			foo353()
			foo354()
			foo355()
			foo356()
			foo357()
			foo358()
			foo359()
			foo360()
			foo361()
			foo362()
			foo363()
			foo364()
			foo365()
			foo366()
			foo367()
			foo368()
			foo369()
			foo370()
			foo371()
			foo372()
			foo373()
			foo374()
			foo375()
			foo376()
			foo377()
			foo378()
			foo379()
			foo380()
			foo381()
			foo382()
			foo383()
			foo384()
			foo385()
			foo386()
			foo387()
			foo388()
			foo389()
			foo390()
			foo391()
			foo392()
			foo393()
			foo394()
			foo395()
			foo396()
			foo397()
			foo398()
			foo399()
			foo400()
			foo401()
			foo402()
			foo403()
			foo404()
			foo405()
			foo406()
			foo407()
			foo408()
			foo409()
			foo410()
			foo411()
			foo412()
			foo413()
			foo414()
			foo415()
			foo416()
			foo417()
			foo418()
			foo419()
			foo420()
			foo421()
			foo422()
			foo423()
			foo424()
			foo425()
			foo426()
			foo427()
			foo428()
			foo429()
			foo430()
			foo431()
			foo432()
			foo433()
			foo434()
			foo435()
			foo436()
			foo437()
			foo438()
			foo439()
			foo440()
			foo441()
			foo442()
			foo443()
			foo444()
			foo445()
			foo446()
			foo447()
			foo448()
			foo449()
			foo450()
			foo451()
			foo452()
			foo453()
			foo454()
			foo455()
			foo456()
			foo457()
			foo458()
			foo459()
			foo460()
			foo461()
			foo462()
			foo463()
			foo464()
			foo465()
			foo466()
			foo467()
			foo468()
			foo469()
			foo470()
			foo471()
			foo472()
			foo473()
			foo474()
			foo475()
			foo476()
			foo477()
			foo478()
			foo479()
			foo480()
			foo481()
			foo482()
			foo483()
			foo484()
			foo485()
			foo486()
			foo487()
			foo488()
			foo489()
			foo490()
			foo491()
			foo492()
			foo493()
			foo494()
			foo495()
			foo496()
			foo497()
			foo498()
			foo499()
			foo500()
			foo501()
			foo502()
			foo503()
			foo504()
			foo505()
			foo506()
			foo507()
			foo508()
			foo509()
			foo510()
			foo511()
		}
		func $(i I) {
			i.foo511()
		}
		`,
		pos: []string{"TESTB"},
	},
	{
		fn: `
		func $(i I) {
			i.foo001()
		}
		`,
		neg: []string{"TESTB"},
	},
}

var linux386Tests = []*asmTest{
	{
		fn: `
		func f0(b []byte) uint32 {
			return binary.LittleEndian.Uint32(b)
		}
		`,
		pos: []string{"\tMOVL\t\\(.*\\),"},
	},
	{
		fn: `
		func f1(b []byte, i int) uint32 {
			return binary.LittleEndian.Uint32(b[i:])
		}
		`,
		pos: []string{"\tMOVL\t\\(.*\\)\\(.*\\*1\\),"},
	},

	// multiplication by powers of two
	{
		fn: `
		func $(n int) int {
			return 32*n
		}
		`,
		pos: []string{"SHLL"},
		neg: []string{"IMULL"},
	},
	{
		fn: `
		func $(n int) int {
			return -64*n
		}
		`,
		pos: []string{"SHLL"},
		neg: []string{"IMULL"},
	},

	// multiplication merging tests
	{
		fn: `
		func $(n int) int {
			return 9*n + 14*n
		}`,
		pos: []string{"\tIMULL\t[$]23"}, // 23*n
	},
	{
		fn: `
		func $(a, n int) int {
			return 19*a + a*n
		}`,
		pos: []string{"\tADDL\t[$]19", "\tIMULL"}, // (n+19)*a
	},
	{
		// check that stack store is optimized away
		fn: `
		func $() int {
			var x int
			return *(&x)
		}
		`,
		pos: []string{"TEXT\t.*, [$]0-4"},
	},
	{
		fn: `
		func mul3(n int) int {
			return 23*n - 9*n
		}`,
		pos: []string{"\tIMULL\t[$]14"}, // 14*n
	},
	{
		fn: `
		func mul4(a, n int) int {
			return n*a - a*19
		}`,
		pos: []string{"\tADDL\t[$]-19", "\tIMULL"}, // (n-19)*a
	},
	// Check that len() and cap() div by a constant power of two
	// are compiled into SHRL.
	{
		fn: `
		func $(a []int) int {
			return len(a) / 1024
		}
		`,
		pos: []string{"\tSHRL\t\\$10,"},
	},
	{
		fn: `
		func $(s string) int {
			return len(s) / (4097 >> 1)
		}
		`,
		pos: []string{"\tSHRL\t\\$11,"},
	},
	{
		fn: `
		func $(a []int) int {
			return cap(a) / ((1 << 11) + 2048)
		}
		`,
		pos: []string{"\tSHRL\t\\$12,"},
	},
	// Check that len() and cap() mod by a constant power of two
	// are compiled into ANDL.
	{
		fn: `
		func $(a []int) int {
			return len(a) % 1024
		}
		`,
		pos: []string{"\tANDL\t\\$1023,"},
	},
	{
		fn: `
		func $(s string) int {
			return len(s) % (4097 >> 1)
		}
		`,
		pos: []string{"\tANDL\t\\$2047,"},
	},
	{
		fn: `
		func $(a []int) int {
			return cap(a) % ((1 << 11) + 2048)
		}
		`,
		pos: []string{"\tANDL\t\\$4095,"},
	},
	{
		// Test that small memmove was replaced with direct movs
		fn: `
                func $() {
                       x := [...]byte{1, 2, 3, 4, 5, 6, 7}
                       copy(x[1:], x[:])
                }
		`,
		neg: []string{"memmove"},
	},
	{
		// Same as above but with different size
		fn: `
                func $() {
                       x := [...]byte{1, 2, 3, 4}
                       copy(x[1:], x[:])
                }
		`,
		neg: []string{"memmove"},
	},
}

var linuxS390XTests = []*asmTest{
	{
		fn: `
		func f0(b []byte) uint32 {
			return binary.LittleEndian.Uint32(b)
		}
		`,
		pos: []string{"\tMOVWBR\t\\(.*\\),"},
	},
	{
		fn: `
		func f1(b []byte, i int) uint32 {
			return binary.LittleEndian.Uint32(b[i:])
		}
		`,
		pos: []string{"\tMOVWBR\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{
		fn: `
		func f2(b []byte) uint64 {
			return binary.LittleEndian.Uint64(b)
		}
		`,
		pos: []string{"\tMOVDBR\t\\(.*\\),"},
	},
	{
		fn: `
		func f3(b []byte, i int) uint64 {
			return binary.LittleEndian.Uint64(b[i:])
		}
		`,
		pos: []string{"\tMOVDBR\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{
		fn: `
		func f4(b []byte) uint32 {
			return binary.BigEndian.Uint32(b)
		}
		`,
		pos: []string{"\tMOVWZ\t\\(.*\\),"},
	},
	{
		fn: `
		func f5(b []byte, i int) uint32 {
			return binary.BigEndian.Uint32(b[i:])
		}
		`,
		pos: []string{"\tMOVWZ\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{
		fn: `
		func f6(b []byte) uint64 {
			return binary.BigEndian.Uint64(b)
		}
		`,
		pos: []string{"\tMOVD\t\\(.*\\),"},
	},
	{
		fn: `
		func f7(b []byte, i int) uint64 {
			return binary.BigEndian.Uint64(b[i:])
		}
		`,
		pos: []string{"\tMOVD\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{
		fn: `
		func f8(x uint64) uint64 {
			return x<<7 + x>>57
		}
		`,
		pos: []string{"\tRLLG\t[$]7,"},
	},
	{
		fn: `
		func f9(x uint64) uint64 {
			return x<<7 | x>>57
		}
		`,
		pos: []string{"\tRLLG\t[$]7,"},
	},
	{
		fn: `
		func f10(x uint64) uint64 {
			return x<<7 ^ x>>57
		}
		`,
		pos: []string{"\tRLLG\t[$]7,"},
	},
	{
		fn: `
		func f11(x uint32) uint32 {
			return x<<7 + x>>25
		}
		`,
		pos: []string{"\tRLL\t[$]7,"},
	},
	{
		fn: `
		func f12(x uint32) uint32 {
			return x<<7 | x>>25
		}
		`,
		pos: []string{"\tRLL\t[$]7,"},
	},
	{
		fn: `
		func f13(x uint32) uint32 {
			return x<<7 ^ x>>25
		}
		`,
		pos: []string{"\tRLL\t[$]7,"},
	},
	// Fused multiply-add/sub instructions.
	{
		fn: `
		func f14(x, y, z float64) float64 {
			return x * y + z
		}
		`,
		pos: []string{"\tFMADD\t"},
	},
	{
		fn: `
		func f15(x, y, z float64) float64 {
			return x * y - z
		}
		`,
		pos: []string{"\tFMSUB\t"},
	},
	{
		fn: `
		func f16(x, y, z float32) float32 {
			return x * y + z
		}
		`,
		pos: []string{"\tFMADDS\t"},
	},
	{
		fn: `
		func f17(x, y, z float32) float32 {
			return x * y - z
		}
		`,
		pos: []string{"\tFMSUBS\t"},
	},
	// Intrinsic tests for math/bits
	{
		fn: `
		func f18(a uint64) int {
			return bits.TrailingZeros64(a)
		}
		`,
		pos: []string{"\tFLOGR\t"},
	},
	{
		fn: `
		func f19(a uint32) int {
			return bits.TrailingZeros32(a)
		}
		`,
		pos: []string{"\tFLOGR\t", "\tMOVWZ\t"},
	},
	{
		fn: `
		func f20(a uint16) int {
			return bits.TrailingZeros16(a)
		}
		`,
		pos: []string{"\tFLOGR\t", "\tOR\t\\$65536,"},
	},
	{
		fn: `
		func f21(a uint8) int {
			return bits.TrailingZeros8(a)
		}
		`,
		pos: []string{"\tFLOGR\t", "\tOR\t\\$256,"},
	},
	// Intrinsic tests for math/bits
	{
		fn: `
		func f22(a uint64) uint64 {
			return bits.ReverseBytes64(a)
		}
		`,
		pos: []string{"\tMOVDBR\t"},
	},
	{
		fn: `
		func f23(a uint32) uint32 {
			return bits.ReverseBytes32(a)
		}
		`,
		pos: []string{"\tMOVWBR\t"},
	},
	{
		fn: `
		func f24(a uint64) int {
			return bits.Len64(a)
		}
		`,
		pos: []string{"\tFLOGR\t"},
	},
	{
		fn: `
		func f25(a uint32) int {
			return bits.Len32(a)
		}
		`,
		pos: []string{"\tFLOGR\t"},
	},
	{
		fn: `
		func f26(a uint16) int {
			return bits.Len16(a)
		}
		`,
		pos: []string{"\tFLOGR\t"},
	},
	{
		fn: `
		func f27(a uint8) int {
			return bits.Len8(a)
		}
		`,
		pos: []string{"\tFLOGR\t"},
	},
	{
		fn: `
		func f28(a uint) int {
			return bits.Len(a)
		}
		`,
		pos: []string{"\tFLOGR\t"},
	},
	{
		fn: `
		func f29(a uint64) int {
			return bits.LeadingZeros64(a)
		}
		`,
		pos: []string{"\tFLOGR\t"},
	},
	{
		fn: `
		func f30(a uint32) int {
			return bits.LeadingZeros32(a)
		}
		`,
		pos: []string{"\tFLOGR\t"},
	},
	{
		fn: `
		func f31(a uint16) int {
			return bits.LeadingZeros16(a)
		}
		`,
		pos: []string{"\tFLOGR\t"},
	},
	{
		fn: `
		func f32(a uint8) int {
			return bits.LeadingZeros8(a)
		}
		`,
		pos: []string{"\tFLOGR\t"},
	},
	{
		fn: `
		func f33(a uint) int {
			return bits.LeadingZeros(a)
		}
		`,
		pos: []string{"\tFLOGR\t"},
	},
	// Intrinsic tests for math.
	{
		fn: `
		func ceil(x float64) float64 {
			return math.Ceil(x)
		}
		`,
		pos: []string{"\tFIDBR\t[$]6"},
	},
	{
		fn: `
		func floor(x float64) float64 {
			return math.Floor(x)
		}
		`,
		pos: []string{"\tFIDBR\t[$]7"},
	},
	{
		fn: `
		func round(x float64) float64 {
			return math.Round(x)
		}
		`,
		pos: []string{"\tFIDBR\t[$]1"},
	},
	{
		fn: `
		func trunc(x float64) float64 {
			return math.Trunc(x)
		}
		`,
		pos: []string{"\tFIDBR\t[$]5"},
	},
	{
		fn: `
		func roundToEven(x float64) float64 {
			return math.RoundToEven(x)
		}
		`,
		pos: []string{"\tFIDBR\t[$]4"},
	},
	{
		// check that stack store is optimized away
		fn: `
		func $() int {
			var x int
			return *(&x)
		}
		`,
		pos: []string{"TEXT\t.*, [$]0-8"},
	},
	// Constant propagation through raw bits conversions.
	{
		// uint32 constant converted to float32 constant
		fn: `
		func $(x float32) float32 {
			if x > math.Float32frombits(0x3f800000) {
				return -x
			}
			return x
		}
		`,
		pos: []string{"\tFMOVS\t[$]f32.3f800000\\(SB\\)"},
	},
	{
		// float32 constant converted to uint32 constant
		fn: `
		func $(x uint32) uint32 {
			if x > math.Float32bits(1) {
				return -x
			}
			return x
		}
		`,
		neg: []string{"\tFMOVS\t"},
	},
	// Constant propagation through float comparisons.
	{
		fn: `
		func $() bool {
			return 0.5 == float64(uint32(1)) ||
				1.5 > float64(uint64(1<<63)) ||
				math.NaN() == math.NaN()
		}
		`,
		pos: []string{"\tMOV(B|BZ|D)\t[$]0,"},
		neg: []string{"\tFCMPU\t", "\tMOV(B|BZ|D)\t[$]1,"},
	},
	{
		fn: `
		func $() bool {
			return float32(0.5) <= float32(int64(1)) &&
				float32(1.5) >= float32(int32(-1<<31)) &&
				float32(math.NaN()) != float32(math.NaN())
		}
		`,
		pos: []string{"\tMOV(B|BZ|D)\t[$]1,"},
		neg: []string{"\tCEBR\t", "\tMOV(B|BZ|D)\t[$]0,"},
	},
	// math tests
	{
		fn: `
		func $(x float64) float64 {
			return math.Abs(x)
		}
		`,
		pos: []string{"\tLPDFR\t"},
		neg: []string{"\tMOVD\t"}, // no integer loads/stores
	},
	{
		fn: `
		func $(x float32) float32 {
			return float32(math.Abs(float64(x)))
		}
		`,
		pos: []string{"\tLPDFR\t"},
		neg: []string{"\tLDEBR\t", "\tLEDBR\t"}, // no float64 conversion
	},
	{
		fn: `
		func $(x float64) float64 {
			return math.Float64frombits(math.Float64bits(x)|1<<63)
		}
		`,
		pos: []string{"\tLNDFR\t"},
		neg: []string{"\tMOVD\t"}, // no integer loads/stores
	},
	{
		fn: `
		func $(x float64) float64 {
			return -math.Abs(x)
		}
		`,
		pos: []string{"\tLNDFR\t"},
		neg: []string{"\tMOVD\t"}, // no integer loads/stores
	},
	{
		fn: `
		func $(x, y float64) float64 {
			return math.Copysign(x, y)
		}
		`,
		pos: []string{"\tCPSDR\t"},
		neg: []string{"\tMOVD\t"}, // no integer loads/stores
	},
	{
		fn: `
		func $(x float64) float64 {
			return math.Copysign(x, -1)
		}
		`,
		pos: []string{"\tLNDFR\t"},
		neg: []string{"\tMOVD\t"}, // no integer loads/stores
	},
	{
		fn: `
		func $(x float64) float64 {
			return math.Copysign(-1, x)
		}
		`,
		pos: []string{"\tCPSDR\t"},
		neg: []string{"\tMOVD\t"}, // no integer loads/stores
	},
}

var linuxARMTests = []*asmTest{
	// multiplication by powers of two
	{
		fn: `
		func $(n int) int {
			return 16*n
		}
		`,
		pos: []string{"\tSLL\t[$]4"},
		neg: []string{"\tMUL\t"},
	},
	{
		fn: `
		func $(n int) int {
			return -32*n
		}
		`,
		pos: []string{"\tSLL\t[$]5"},
		neg: []string{"\tMUL\t"},
	},

	{
		fn: `
		func f0(x uint32) uint32 {
			return x<<7 + x>>25
		}
		`,
		pos: []string{"\tMOVW\tR[0-9]+@>25,"},
	},
	{
		fn: `
		func f1(x uint32) uint32 {
			return x<<7 | x>>25
		}
		`,
		pos: []string{"\tMOVW\tR[0-9]+@>25,"},
	},
	{
		fn: `
		func f2(x uint32) uint32 {
			return x<<7 ^ x>>25
		}
		`,
		pos: []string{"\tMOVW\tR[0-9]+@>25,"},
	},
	{
		fn: `
		func f3(a uint64) int {
			return bits.Len64(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f4(a uint32) int {
			return bits.Len32(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f5(a uint16) int {
			return bits.Len16(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f6(a uint8) int {
			return bits.Len8(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f7(a uint) int {
			return bits.Len(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f8(a uint64) int {
			return bits.LeadingZeros64(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f9(a uint32) int {
			return bits.LeadingZeros32(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f10(a uint16) int {
			return bits.LeadingZeros16(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f11(a uint8) int {
			return bits.LeadingZeros8(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f12(a uint) int {
			return bits.LeadingZeros(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
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
	{
		// check that stack store is optimized away
		fn: `
		func $() int {
			var x int
			return *(&x)
		}
		`,
		pos: []string{"TEXT\t.*, [$]-4-4"},
	},
}

var linuxARM64Tests = []*asmTest{
	// multiplication by powers of two
	{
		fn: `
		func $(n int) int {
			return 64*n
		}
		`,
		pos: []string{"\tLSL\t[$]6"},
		neg: []string{"\tMUL\t"},
	},
	{
		fn: `
		func $(n int) int {
			return -128*n
		}
		`,
		pos: []string{"\tLSL\t[$]7"},
		neg: []string{"\tMUL\t"},
	},

	{
		fn: `
		func f0(x uint64) uint64 {
			return x<<7 + x>>57
		}
		`,
		pos: []string{"\tROR\t[$]57,"},
	},
	{
		fn: `
		func f1(x uint64) uint64 {
			return x<<7 | x>>57
		}
		`,
		pos: []string{"\tROR\t[$]57,"},
	},
	{
		fn: `
		func f2(x uint64) uint64 {
			return x<<7 ^ x>>57
		}
		`,
		pos: []string{"\tROR\t[$]57,"},
	},
	{
		fn: `
		func f3(x uint32) uint32 {
			return x<<7 + x>>25
		}
		`,
		pos: []string{"\tRORW\t[$]25,"},
	},
	{
		fn: `
		func f4(x uint32) uint32 {
			return x<<7 | x>>25
		}
		`,
		pos: []string{"\tRORW\t[$]25,"},
	},
	{
		fn: `
		func f5(x uint32) uint32 {
			return x<<7 ^ x>>25
		}
		`,
		pos: []string{"\tRORW\t[$]25,"},
	},
	{
		fn: `
		func f22(a uint64) uint64 {
			return bits.ReverseBytes64(a)
		}
		`,
		pos: []string{"\tREV\t"},
	},
	{
		fn: `
		func f23(a uint32) uint32 {
			return bits.ReverseBytes32(a)
		}
		`,
		pos: []string{"\tREVW\t"},
	},
	{
		fn: `
		func f24(a uint64) int {
			return bits.Len64(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f25(a uint32) int {
			return bits.Len32(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f26(a uint16) int {
			return bits.Len16(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f27(a uint8) int {
			return bits.Len8(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f28(a uint) int {
			return bits.Len(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f29(a uint64) int {
			return bits.LeadingZeros64(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f30(a uint32) int {
			return bits.LeadingZeros32(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f31(a uint16) int {
			return bits.LeadingZeros16(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f32(a uint8) int {
			return bits.LeadingZeros8(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f33(a uint) int {
			return bits.LeadingZeros(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
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
		// check that stack store is optimized away
		fn: `
		func $() int {
			var x int
			return *(&x)
		}
		`,
		pos: []string{"TEXT\t.*, [$]-8-8"},
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
}

var linuxMIPSTests = []*asmTest{
	{
		fn: `
		func f0(a uint64) int {
			return bits.Len64(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f1(a uint32) int {
			return bits.Len32(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f2(a uint16) int {
			return bits.Len16(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f3(a uint8) int {
			return bits.Len8(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f4(a uint) int {
			return bits.Len(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f5(a uint64) int {
			return bits.LeadingZeros64(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f6(a uint32) int {
			return bits.LeadingZeros32(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f7(a uint16) int {
			return bits.LeadingZeros16(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f8(a uint8) int {
			return bits.LeadingZeros8(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		fn: `
		func f9(a uint) int {
			return bits.LeadingZeros(a)
		}
		`,
		pos: []string{"\tCLZ\t"},
	},
	{
		// check that stack store is optimized away
		fn: `
		func $() int {
			var x int
			return *(&x)
		}
		`,
		pos: []string{"TEXT\t.*, [$]-4-4"},
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

var linuxPPC64LETests = []*asmTest{
	// Fused multiply-add/sub instructions.
	{
		fn: `
		func f0(x, y, z float64) float64 {
			return x * y + z
		}
		`,
		pos: []string{"\tFMADD\t"},
	},
	{
		fn: `
		func f1(x, y, z float64) float64 {
			return x * y - z
		}
		`,
		pos: []string{"\tFMSUB\t"},
	},
	{
		fn: `
		func f2(x, y, z float32) float32 {
			return x * y + z
		}
		`,
		pos: []string{"\tFMADDS\t"},
	},
	{
		fn: `
		func f3(x, y, z float32) float32 {
			return x * y - z
		}
		`,
		pos: []string{"\tFMSUBS\t"},
	},
	{
		fn: `
		func f4(x uint32) uint32 {
			return x<<7 | x>>25
		}
		`,
		pos: []string{"\tROTLW\t"},
	},
	{
		fn: `
		func f5(x uint32) uint32 {
			return x<<7 + x>>25
		}
		`,
		pos: []string{"\tROTLW\t"},
	},
	{
		fn: `
		func f6(x uint32) uint32 {
			return x<<7 ^ x>>25
		}
		`,
		pos: []string{"\tROTLW\t"},
	},
	{
		fn: `
		func f7(x uint64) uint64 {
			return x<<7 | x>>57
		}
		`,
		pos: []string{"\tROTL\t"},
	},
	{
		fn: `
		func f8(x uint64) uint64 {
			return x<<7 + x>>57
		}
		`,
		pos: []string{"\tROTL\t"},
	},
	{
		fn: `
		func f9(x uint64) uint64 {
			return x<<7 ^ x>>57
		}
		`,
		pos: []string{"\tROTL\t"},
	},
	{
		fn: `
		func f10(a uint32) uint32 {
			return bits.RotateLeft32(a, 9)
		}
		`,
		pos: []string{"\tROTLW\t"},
	},
	{
		fn: `
		func f11(a uint64) uint64 {
			return bits.RotateLeft64(a, 37)
		}
		`,
		pos: []string{"\tROTL\t"},
	},

	{
		fn: `
                func f12(a, b float64) float64 {
                        return math.Copysign(a, b)
                }
                `,
		pos: []string{"\tFCPSGN\t"},
	},

	{
		fn: `
                func f13(a float64) float64 {
                        return math.Abs(a)
                }
                `,
		pos: []string{"\tFABS\t"},
	},

	{
		fn: `
		func f14(b []byte) uint16 {
			return binary.LittleEndian.Uint16(b)
	}
		`,
		pos: []string{"\tMOVHZ\t"},
	},
	{
		fn: `
		func f15(b []byte) uint32 {
			return binary.LittleEndian.Uint32(b)
		}
		`,
		pos: []string{"\tMOVWZ\t"},
	},

	{
		fn: `
		func f16(b []byte) uint64 {
			return binary.LittleEndian.Uint64(b)
		}
		`,
		pos: []string{"\tMOVD\t"},
		neg: []string{"MOVBZ", "MOVHZ", "MOVWZ"},
	},

	{
		fn: `
		func f17(b []byte, v uint16) {
			binary.LittleEndian.PutUint16(b, v)
		}
		`,
		pos: []string{"\tMOVH\t"},
	},

	{
		fn: `
		func f18(b []byte, v uint32) {
			binary.LittleEndian.PutUint32(b, v)
		}
		`,
		pos: []string{"\tMOVW\t"},
	},

	{
		fn: `
		func f19(b []byte, v uint64) {
			binary.LittleEndian.PutUint64(b, v)
		}
		`,
		pos: []string{"\tMOVD\t"},
		neg: []string{"MOVB", "MOVH", "MOVW"},
	},

	{
		// check that stack store is optimized away
		fn: `
		func $() int {
			var x int
			return *(&x)
		}
		`,
		pos: []string{"TEXT\t.*, [$]0-8"},
	},
	// Constant propagation through raw bits conversions.
	{
		// uint32 constant converted to float32 constant
		fn: `
		func $(x float32) float32 {
			if x > math.Float32frombits(0x3f800000) {
				return -x
			}
			return x
		}
		`,
		pos: []string{"\tFMOVS\t[$]f32.3f800000\\(SB\\)"},
	},
	{
		// float32 constant converted to uint32 constant
		fn: `
		func $(x uint32) uint32 {
			if x > math.Float32bits(1) {
				return -x
			}
			return x
		}
		`,
		neg: []string{"\tFMOVS\t"},
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
