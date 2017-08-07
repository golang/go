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

				for _, at := range ats.tests {
					funcName := nameRegexp.FindString(at.function)[len("func "):]
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
	// function to compile, must be named fX,
	// where X is this test's index in asmTests.tests.
	function string
	// regexps that must match the generated assembly
	regexps []string
}

func (at asmTest) verifyAsm(t *testing.T, fa string) {
	for _, r := range at.regexps {
		if b, err := regexp.MatchString(r, fa); !b || err != nil {
			t.Errorf("expected:%s\ngo:%s\nasm:%s\n", r, at.function, fa)
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

	for _, t := range ats.tests {
		fmt.Fprintln(&buf, t.function)
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
		imports: []string{"encoding/binary", "math/bits", "unsafe"},
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
		imports: []string{"encoding/binary", "math/bits"},
		tests:   linuxS390XTests,
	},
	{
		arch:    "arm",
		os:      "linux",
		imports: []string{"math/bits"},
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
		arch:  "ppc64le",
		os:    "linux",
		tests: linuxPPC64LETests,
	},
}

var linuxAMD64Tests = []*asmTest{
	{
		`
		func f0(x int) int {
			return x * 64
		}
		`,
		[]string{"\tSHLQ\t\\$6,"},
	},
	{
		`
		func f1(x int) int {
			return x * 96
		}
		`,
		[]string{"\tSHLQ\t\\$5,", "\tLEAQ\t\\(.*\\)\\(.*\\*2\\),"},
	},
	// Load-combining tests.
	{
		`
		func f2(b []byte) uint64 {
			return binary.LittleEndian.Uint64(b)
		}
		`,
		[]string{"\tMOVQ\t\\(.*\\),"},
	},
	{
		`
		func f3(b []byte, i int) uint64 {
			return binary.LittleEndian.Uint64(b[i:])
		}
		`,
		[]string{"\tMOVQ\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{
		`
		func f4(b []byte) uint32 {
			return binary.LittleEndian.Uint32(b)
		}
		`,
		[]string{"\tMOVL\t\\(.*\\),"},
	},
	{
		`
		func f5(b []byte, i int) uint32 {
			return binary.LittleEndian.Uint32(b[i:])
		}
		`,
		[]string{"\tMOVL\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{
		`
		func f6(b []byte) uint64 {
			return binary.BigEndian.Uint64(b)
		}
		`,
		[]string{"\tBSWAPQ\t"},
	},
	{
		`
		func f7(b []byte, i int) uint64 {
			return binary.BigEndian.Uint64(b[i:])
		}
		`,
		[]string{"\tBSWAPQ\t"},
	},
	{
		`
		func f8(b []byte, v uint64) {
			binary.BigEndian.PutUint64(b, v)
		}
		`,
		[]string{"\tBSWAPQ\t"},
	},
	{
		`
		func f9(b []byte, i int, v uint64) {
			binary.BigEndian.PutUint64(b[i:], v)
		}
		`,
		[]string{"\tBSWAPQ\t"},
	},
	{
		`
		func f10(b []byte) uint32 {
			return binary.BigEndian.Uint32(b)
		}
		`,
		[]string{"\tBSWAPL\t"},
	},
	{
		`
		func f11(b []byte, i int) uint32 {
			return binary.BigEndian.Uint32(b[i:])
		}
		`,
		[]string{"\tBSWAPL\t"},
	},
	{
		`
		func f12(b []byte, v uint32) {
			binary.BigEndian.PutUint32(b, v)
		}
		`,
		[]string{"\tBSWAPL\t"},
	},
	{
		`
		func f13(b []byte, i int, v uint32) {
			binary.BigEndian.PutUint32(b[i:], v)
		}
		`,
		[]string{"\tBSWAPL\t"},
	},
	{
		`
		func f14(b []byte) uint16 {
			return binary.BigEndian.Uint16(b)
		}
		`,
		[]string{"\tROLW\t\\$8,"},
	},
	{
		`
		func f15(b []byte, i int) uint16 {
			return binary.BigEndian.Uint16(b[i:])
		}
		`,
		[]string{"\tROLW\t\\$8,"},
	},
	{
		`
		func f16(b []byte, v uint16) {
			binary.BigEndian.PutUint16(b, v)
		}
		`,
		[]string{"\tROLW\t\\$8,"},
	},
	{
		`
		func f17(b []byte, i int, v uint16) {
			binary.BigEndian.PutUint16(b[i:], v)
		}
		`,
		[]string{"\tROLW\t\\$8,"},
	},
	// Structure zeroing.  See issue #18370.
	{
		`
		type T1 struct {
			a, b, c int
		}
		func f18(t *T1) {
			*t = T1{}
		}
		`,
		[]string{"\tMOVQ\t\\$0, \\(.*\\)", "\tMOVQ\t\\$0, 8\\(.*\\)", "\tMOVQ\t\\$0, 16\\(.*\\)"},
	},
	// SSA-able composite literal initialization. Issue 18872.
	{
		`
		type T18872 struct {
			a, b, c, d int
		}

		func f18872(p *T18872) {
			*p = T18872{1, 2, 3, 4}
		}
		`,
		[]string{"\tMOVQ\t[$]1", "\tMOVQ\t[$]2", "\tMOVQ\t[$]3", "\tMOVQ\t[$]4"},
	},
	// Also test struct containing pointers (this was special because of write barriers).
	{
		`
		type T2 struct {
			a, b, c *int
		}
		func f19(t *T2) {
			*t = T2{}
		}
		`,
		[]string{"\tMOVQ\t\\$0, \\(.*\\)", "\tMOVQ\t\\$0, 8\\(.*\\)", "\tMOVQ\t\\$0, 16\\(.*\\)", "\tCALL\truntime\\.writebarrierptr\\(SB\\)"},
	},
	// Rotate tests
	{
		`
		func f20(x uint64) uint64 {
			return x<<7 | x>>57
		}
		`,
		[]string{"\tROLQ\t[$]7,"},
	},
	{
		`
		func f21(x uint64) uint64 {
			return x<<7 + x>>57
		}
		`,
		[]string{"\tROLQ\t[$]7,"},
	},
	{
		`
		func f22(x uint64) uint64 {
			return x<<7 ^ x>>57
		}
		`,
		[]string{"\tROLQ\t[$]7,"},
	},
	{
		`
		func f23(x uint32) uint32 {
			return x<<7 + x>>25
		}
		`,
		[]string{"\tROLL\t[$]7,"},
	},
	{
		`
		func f24(x uint32) uint32 {
			return x<<7 | x>>25
		}
		`,
		[]string{"\tROLL\t[$]7,"},
	},
	{
		`
		func f25(x uint32) uint32 {
			return x<<7 ^ x>>25
		}
		`,
		[]string{"\tROLL\t[$]7,"},
	},
	{
		`
		func f26(x uint16) uint16 {
			return x<<7 + x>>9
		}
		`,
		[]string{"\tROLW\t[$]7,"},
	},
	{
		`
		func f27(x uint16) uint16 {
			return x<<7 | x>>9
		}
		`,
		[]string{"\tROLW\t[$]7,"},
	},
	{
		`
		func f28(x uint16) uint16 {
			return x<<7 ^ x>>9
		}
		`,
		[]string{"\tROLW\t[$]7,"},
	},
	{
		`
		func f29(x uint8) uint8 {
			return x<<7 + x>>1
		}
		`,
		[]string{"\tROLB\t[$]7,"},
	},
	{
		`
		func f30(x uint8) uint8 {
			return x<<7 | x>>1
		}
		`,
		[]string{"\tROLB\t[$]7,"},
	},
	{
		`
		func f31(x uint8) uint8 {
			return x<<7 ^ x>>1
		}
		`,
		[]string{"\tROLB\t[$]7,"},
	},
	// Rotate after inlining (see issue 18254).
	{
		`
		func f32(x uint32) uint32 {
			return g(x, 7)
		}
		func g(x uint32, k uint) uint32 {
			return x<<k | x>>(32-k)
		}
		`,
		[]string{"\tROLL\t[$]7,"},
	},
	{
		`
		func f33(m map[int]int) int {
			return m[5]
		}
		`,
		[]string{"\tMOVQ\t[$]5,"},
	},
	// Direct use of constants in fast map access calls. Issue 19015.
	{
		`
		func f34(m map[int]int) bool {
			_, ok := m[5]
			return ok
		}
		`,
		[]string{"\tMOVQ\t[$]5,"},
	},
	{
		`
		func f35(m map[string]int) int {
			return m["abc"]
		}
		`,
		[]string{"\"abc\""},
	},
	{
		`
		func f36(m map[string]int) bool {
			_, ok := m["abc"]
			return ok
		}
		`,
		[]string{"\"abc\""},
	},
	// Bit test ops on amd64, issue 18943.
	{
		`
		func f37(a, b uint64) int {
			if a&(1<<(b&63)) != 0 {
				return 1
			}
			return -1
		}
		`,
		[]string{"\tBTQ\t"},
	},
	{
		`
		func f38(a, b uint64) bool {
			return a&(1<<(b&63)) != 0
		}
		`,
		[]string{"\tBTQ\t"},
	},
	{
		`
		func f39(a uint64) int {
			if a&(1<<60) != 0 {
				return 1
			}
			return -1
		}
		`,
		[]string{"\tBTQ\t\\$60"},
	},
	{
		`
		func f40(a uint64) bool {
			return a&(1<<60) != 0
		}
		`,
		[]string{"\tBTQ\t\\$60"},
	},
	// Intrinsic tests for math/bits
	{
		`
		func f41(a uint64) int {
			return bits.TrailingZeros64(a)
		}
		`,
		[]string{"\tBSFQ\t", "\tMOVL\t\\$64,", "\tCMOVQEQ\t"},
	},
	{
		`
		func f42(a uint32) int {
			return bits.TrailingZeros32(a)
		}
		`,
		[]string{"\tBSFQ\t", "\tORQ\t[^$]", "\tMOVQ\t\\$4294967296,"},
	},
	{
		`
		func f43(a uint16) int {
			return bits.TrailingZeros16(a)
		}
		`,
		[]string{"\tBSFQ\t", "\tORQ\t\\$65536,"},
	},
	{
		`
		func f44(a uint8) int {
			return bits.TrailingZeros8(a)
		}
		`,
		[]string{"\tBSFQ\t", "\tORQ\t\\$256,"},
	},
	{
		`
		func f45(a uint64) uint64 {
			return bits.ReverseBytes64(a)
		}
		`,
		[]string{"\tBSWAPQ\t"},
	},
	{
		`
		func f46(a uint32) uint32 {
			return bits.ReverseBytes32(a)
		}
		`,
		[]string{"\tBSWAPL\t"},
	},
	{
		`
		func f47(a uint16) uint16 {
			return bits.ReverseBytes16(a)
		}
		`,
		[]string{"\tROLW\t\\$8,"},
	},
	{
		`
		func f48(a uint64) int {
			return bits.Len64(a)
		}
		`,
		[]string{"\tBSRQ\t"},
	},
	{
		`
		func f49(a uint32) int {
			return bits.Len32(a)
		}
		`,
		[]string{"\tBSRQ\t"},
	},
	{
		`
		func f50(a uint16) int {
			return bits.Len16(a)
		}
		`,
		[]string{"\tBSRQ\t"},
	},
	/* see ssa.go
	{
		`
		func f51(a uint8) int {
			return bits.Len8(a)
		}
		`,
		[]string{"\tBSRQ\t"},
	},
	*/
	{
		`
		func f52(a uint) int {
			return bits.Len(a)
		}
		`,
		[]string{"\tBSRQ\t"},
	},
	{
		`
		func f53(a uint64) int {
			return bits.LeadingZeros64(a)
		}
		`,
		[]string{"\tBSRQ\t"},
	},
	{
		`
		func f54(a uint32) int {
			return bits.LeadingZeros32(a)
		}
		`,
		[]string{"\tBSRQ\t"},
	},
	{
		`
		func f55(a uint16) int {
			return bits.LeadingZeros16(a)
		}
		`,
		[]string{"\tBSRQ\t"},
	},
	/* see ssa.go
	{
		`
		func f56(a uint8) int {
			return bits.LeadingZeros8(a)
		}
		`,
		[]string{"\tBSRQ\t"},
	},
	*/
	{
		`
		func f57(a uint) int {
			return bits.LeadingZeros(a)
		}
		`,
		[]string{"\tBSRQ\t"},
	},
	{
		`
		func pop1(x uint64) int {
			return bits.OnesCount64(x)
		}`,
		[]string{"\tPOPCNTQ\t", "support_popcnt"},
	},
	{
		`
		func pop2(x uint32) int {
			return bits.OnesCount32(x)
		}`,
		[]string{"\tPOPCNTL\t", "support_popcnt"},
	},
	{
		`
		func pop3(x uint16) int {
			return bits.OnesCount16(x)
		}`,
		[]string{"\tPOPCNTL\t", "support_popcnt"},
	},
	{
		`
		func pop4(x uint) int {
			return bits.OnesCount(x)
		}`,
		[]string{"\tPOPCNTQ\t", "support_popcnt"},
	},
	// see issue 19595.
	// We want to merge load+op in f58, but not in f59.
	{
		`
		func f58(p, q *int) {
			x := *p
			*q += x
		}`,
		[]string{"\tADDQ\t\\("},
	},
	{
		`
		func f59(p, q *int) {
			x := *p
			for i := 0; i < 10; i++ {
				*q += x
			}
		}`,
		[]string{"\tADDQ\t[A-Z]"},
	},
	// Floating-point strength reduction
	{
		`
		func f60(f float64) float64 {
			return f * 2.0
		}`,
		[]string{"\tADDSD\t"},
	},
	{
		`
		func f62(f float64) float64 {
			return f / 16.0
		}`,
		[]string{"\tMULSD\t"},
	},
	{
		`
		func f63(f float64) float64 {
			return f / 0.125
		}`,
		[]string{"\tMULSD\t"},
	},
	{
		`
		func f64(f float64) float64 {
			return f / 0.5
		}`,
		[]string{"\tADDSD\t"},
	},
	// Check that compare to constant string uses 2/4/8 byte compares
	{
		`
		func f65(a string) bool {
		    return a == "xx"
		}`,
		[]string{"\tCMPW\t[A-Z]"},
	},
	{
		`
		func f66(a string) bool {
		    return a == "xxxx"
		}`,
		[]string{"\tCMPL\t[A-Z]"},
	},
	{
		`
		func f67(a string) bool {
		    return a == "xxxxxxxx"
		}`,
		[]string{"\tCMPQ\t[A-Z]"},
	},
	// Non-constant rotate
	{
		`func rot64l(x uint64, y int) uint64 {
			z := uint(y & 63)
			return x << z | x >> (64-z)
		}`,
		[]string{"\tROLQ\t"},
	},
	{
		`func rot64r(x uint64, y int) uint64 {
			z := uint(y & 63)
			return x >> z | x << (64-z)
		}`,
		[]string{"\tRORQ\t"},
	},
	{
		`func rot32l(x uint32, y int) uint32 {
			z := uint(y & 31)
			return x << z | x >> (32-z)
		}`,
		[]string{"\tROLL\t"},
	},
	{
		`func rot32r(x uint32, y int) uint32 {
			z := uint(y & 31)
			return x >> z | x << (32-z)
		}`,
		[]string{"\tRORL\t"},
	},
	{
		`func rot16l(x uint16, y int) uint16 {
			z := uint(y & 15)
			return x << z | x >> (16-z)
		}`,
		[]string{"\tROLW\t"},
	},
	{
		`func rot16r(x uint16, y int) uint16 {
			z := uint(y & 15)
			return x >> z | x << (16-z)
		}`,
		[]string{"\tRORW\t"},
	},
	{
		`func rot8l(x uint8, y int) uint8 {
			z := uint(y & 7)
			return x << z | x >> (8-z)
		}`,
		[]string{"\tROLB\t"},
	},
	{
		`func rot8r(x uint8, y int) uint8 {
			z := uint(y & 7)
			return x >> z | x << (8-z)
		}`,
		[]string{"\tRORB\t"},
	},
	// Check that array compare uses 2/4/8 byte compares
	{
		`
		func f68(a,b [2]byte) bool {
		    return a == b
		}`,
		[]string{"\tCMPW\t[A-Z]"},
	},
	{
		`
		func f69(a,b [3]uint16) bool {
		    return a == b
		}`,
		[]string{"\tCMPL\t[A-Z]"},
	},
	{
		`
		func f70(a,b [15]byte) bool {
		    return a == b
		}`,
		[]string{"\tCMPQ\t[A-Z]"},
	},
	{
		`
		func f71(a,b unsafe.Pointer) bool { // This was a TODO in mapaccess1_faststr
		    return *((*[4]byte)(a)) != *((*[4]byte)(b))
		}`,
		[]string{"\tCMPL\t[A-Z]"},
	},
	{
		// make sure assembly output has matching offset and base register.
		`
		func f72(a, b int) int {
			var x [16]byte // use some frame
			_ = x
			return b
		}
		`,
		[]string{"b\\+40\\(SP\\)"},
	},
}

var linux386Tests = []*asmTest{
	{
		`
		func f0(b []byte) uint32 {
			return binary.LittleEndian.Uint32(b)
		}
		`,
		[]string{"\tMOVL\t\\(.*\\),"},
	},
	{
		`
		func f1(b []byte, i int) uint32 {
			return binary.LittleEndian.Uint32(b[i:])
		}
		`,
		[]string{"\tMOVL\t\\(.*\\)\\(.*\\*1\\),"},
	},
}

var linuxS390XTests = []*asmTest{
	{
		`
		func f0(b []byte) uint32 {
			return binary.LittleEndian.Uint32(b)
		}
		`,
		[]string{"\tMOVWBR\t\\(.*\\),"},
	},
	{
		`
		func f1(b []byte, i int) uint32 {
			return binary.LittleEndian.Uint32(b[i:])
		}
		`,
		[]string{"\tMOVWBR\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{
		`
		func f2(b []byte) uint64 {
			return binary.LittleEndian.Uint64(b)
		}
		`,
		[]string{"\tMOVDBR\t\\(.*\\),"},
	},
	{
		`
		func f3(b []byte, i int) uint64 {
			return binary.LittleEndian.Uint64(b[i:])
		}
		`,
		[]string{"\tMOVDBR\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{
		`
		func f4(b []byte) uint32 {
			return binary.BigEndian.Uint32(b)
		}
		`,
		[]string{"\tMOVWZ\t\\(.*\\),"},
	},
	{
		`
		func f5(b []byte, i int) uint32 {
			return binary.BigEndian.Uint32(b[i:])
		}
		`,
		[]string{"\tMOVWZ\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{
		`
		func f6(b []byte) uint64 {
			return binary.BigEndian.Uint64(b)
		}
		`,
		[]string{"\tMOVD\t\\(.*\\),"},
	},
	{
		`
		func f7(b []byte, i int) uint64 {
			return binary.BigEndian.Uint64(b[i:])
		}
		`,
		[]string{"\tMOVD\t\\(.*\\)\\(.*\\*1\\),"},
	},
	{
		`
		func f8(x uint64) uint64 {
			return x<<7 + x>>57
		}
		`,
		[]string{"\tRLLG\t[$]7,"},
	},
	{
		`
		func f9(x uint64) uint64 {
			return x<<7 | x>>57
		}
		`,
		[]string{"\tRLLG\t[$]7,"},
	},
	{
		`
		func f10(x uint64) uint64 {
			return x<<7 ^ x>>57
		}
		`,
		[]string{"\tRLLG\t[$]7,"},
	},
	{
		`
		func f11(x uint32) uint32 {
			return x<<7 + x>>25
		}
		`,
		[]string{"\tRLL\t[$]7,"},
	},
	{
		`
		func f12(x uint32) uint32 {
			return x<<7 | x>>25
		}
		`,
		[]string{"\tRLL\t[$]7,"},
	},
	{
		`
		func f13(x uint32) uint32 {
			return x<<7 ^ x>>25
		}
		`,
		[]string{"\tRLL\t[$]7,"},
	},
	// Fused multiply-add/sub instructions.
	{
		`
		func f14(x, y, z float64) float64 {
			return x * y + z
		}
		`,
		[]string{"\tFMADD\t"},
	},
	{
		`
		func f15(x, y, z float64) float64 {
			return x * y - z
		}
		`,
		[]string{"\tFMSUB\t"},
	},
	{
		`
		func f16(x, y, z float32) float32 {
			return x * y + z
		}
		`,
		[]string{"\tFMADDS\t"},
	},
	{
		`
		func f17(x, y, z float32) float32 {
			return x * y - z
		}
		`,
		[]string{"\tFMSUBS\t"},
	},
	// Intrinsic tests for math/bits
	{
		`
		func f18(a uint64) int {
			return bits.TrailingZeros64(a)
		}
		`,
		[]string{"\tFLOGR\t"},
	},
	{
		`
		func f19(a uint32) int {
			return bits.TrailingZeros32(a)
		}
		`,
		[]string{"\tFLOGR\t", "\tMOVWZ\t"},
	},
	{
		`
		func f20(a uint16) int {
			return bits.TrailingZeros16(a)
		}
		`,
		[]string{"\tFLOGR\t", "\tOR\t\\$65536,"},
	},
	{
		`
		func f21(a uint8) int {
			return bits.TrailingZeros8(a)
		}
		`,
		[]string{"\tFLOGR\t", "\tOR\t\\$256,"},
	},
	// Intrinsic tests for math/bits
	{
		`
		func f22(a uint64) uint64 {
			return bits.ReverseBytes64(a)
		}
		`,
		[]string{"\tMOVDBR\t"},
	},
	{
		`
		func f23(a uint32) uint32 {
			return bits.ReverseBytes32(a)
		}
		`,
		[]string{"\tMOVWBR\t"},
	},
	{
		`
		func f24(a uint64) int {
			return bits.Len64(a)
		}
		`,
		[]string{"\tFLOGR\t"},
	},
	{
		`
		func f25(a uint32) int {
			return bits.Len32(a)
		}
		`,
		[]string{"\tFLOGR\t"},
	},
	{
		`
		func f26(a uint16) int {
			return bits.Len16(a)
		}
		`,
		[]string{"\tFLOGR\t"},
	},
	{
		`
		func f27(a uint8) int {
			return bits.Len8(a)
		}
		`,
		[]string{"\tFLOGR\t"},
	},
	{
		`
		func f28(a uint) int {
			return bits.Len(a)
		}
		`,
		[]string{"\tFLOGR\t"},
	},
	{
		`
		func f29(a uint64) int {
			return bits.LeadingZeros64(a)
		}
		`,
		[]string{"\tFLOGR\t"},
	},
	{
		`
		func f30(a uint32) int {
			return bits.LeadingZeros32(a)
		}
		`,
		[]string{"\tFLOGR\t"},
	},
	{
		`
		func f31(a uint16) int {
			return bits.LeadingZeros16(a)
		}
		`,
		[]string{"\tFLOGR\t"},
	},
	{
		`
		func f32(a uint8) int {
			return bits.LeadingZeros8(a)
		}
		`,
		[]string{"\tFLOGR\t"},
	},
	{
		`
		func f33(a uint) int {
			return bits.LeadingZeros(a)
		}
		`,
		[]string{"\tFLOGR\t"},
	},
}

var linuxARMTests = []*asmTest{
	{
		`
		func f0(x uint32) uint32 {
			return x<<7 + x>>25
		}
		`,
		[]string{"\tMOVW\tR[0-9]+@>25,"},
	},
	{
		`
		func f1(x uint32) uint32 {
			return x<<7 | x>>25
		}
		`,
		[]string{"\tMOVW\tR[0-9]+@>25,"},
	},
	{
		`
		func f2(x uint32) uint32 {
			return x<<7 ^ x>>25
		}
		`,
		[]string{"\tMOVW\tR[0-9]+@>25,"},
	},
	{
		`
		func f3(a uint64) int {
			return bits.Len64(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f4(a uint32) int {
			return bits.Len32(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f5(a uint16) int {
			return bits.Len16(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f6(a uint8) int {
			return bits.Len8(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f7(a uint) int {
			return bits.Len(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f8(a uint64) int {
			return bits.LeadingZeros64(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f9(a uint32) int {
			return bits.LeadingZeros32(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f10(a uint16) int {
			return bits.LeadingZeros16(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f11(a uint8) int {
			return bits.LeadingZeros8(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f12(a uint) int {
			return bits.LeadingZeros(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		// make sure assembly output has matching offset and base register.
		`
		func f13(a, b int) int {
			var x [16]byte // use some frame
			_ = x
			return b
		}
		`,
		[]string{"b\\+4\\(FP\\)"},
	},
}

var linuxARM64Tests = []*asmTest{
	{
		`
		func f0(x uint64) uint64 {
			return x<<7 + x>>57
		}
		`,
		[]string{"\tROR\t[$]57,"},
	},
	{
		`
		func f1(x uint64) uint64 {
			return x<<7 | x>>57
		}
		`,
		[]string{"\tROR\t[$]57,"},
	},
	{
		`
		func f2(x uint64) uint64 {
			return x<<7 ^ x>>57
		}
		`,
		[]string{"\tROR\t[$]57,"},
	},
	{
		`
		func f3(x uint32) uint32 {
			return x<<7 + x>>25
		}
		`,
		[]string{"\tRORW\t[$]25,"},
	},
	{
		`
		func f4(x uint32) uint32 {
			return x<<7 | x>>25
		}
		`,
		[]string{"\tRORW\t[$]25,"},
	},
	{
		`
		func f5(x uint32) uint32 {
			return x<<7 ^ x>>25
		}
		`,
		[]string{"\tRORW\t[$]25,"},
	},
	{
		`
		func f22(a uint64) uint64 {
			return bits.ReverseBytes64(a)
		}
		`,
		[]string{"\tREV\t"},
	},
	{
		`
		func f23(a uint32) uint32 {
			return bits.ReverseBytes32(a)
		}
		`,
		[]string{"\tREVW\t"},
	},
	{
		`
		func f24(a uint64) int {
			return bits.Len64(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f25(a uint32) int {
			return bits.Len32(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f26(a uint16) int {
			return bits.Len16(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f27(a uint8) int {
			return bits.Len8(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f28(a uint) int {
			return bits.Len(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f29(a uint64) int {
			return bits.LeadingZeros64(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f30(a uint32) int {
			return bits.LeadingZeros32(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f31(a uint16) int {
			return bits.LeadingZeros16(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f32(a uint8) int {
			return bits.LeadingZeros8(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f33(a uint) int {
			return bits.LeadingZeros(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f34(a uint64) uint64 {
			return a & ((1<<63)-1)
		}
		`,
		[]string{"\tAND\t"},
	},
	{
		`
		func f35(a uint64) uint64 {
			return a & (1<<63)
		}
		`,
		[]string{"\tAND\t"},
	},
	{
		// make sure offsets are folded into load and store.
		`
		func f36(_, a [20]byte) (b [20]byte) {
			b = a
			return
		}
		`,
		[]string{"\tMOVD\t\"\"\\.a\\+[0-9]+\\(FP\\), R[0-9]+", "\tMOVD\tR[0-9]+, \"\"\\.b\\+[0-9]+\\(FP\\)"},
	},
}

var linuxMIPSTests = []*asmTest{
	{
		`
		func f0(a uint64) int {
			return bits.Len64(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f1(a uint32) int {
			return bits.Len32(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f2(a uint16) int {
			return bits.Len16(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f3(a uint8) int {
			return bits.Len8(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f4(a uint) int {
			return bits.Len(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f5(a uint64) int {
			return bits.LeadingZeros64(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f6(a uint32) int {
			return bits.LeadingZeros32(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f7(a uint16) int {
			return bits.LeadingZeros16(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f8(a uint8) int {
			return bits.LeadingZeros8(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
	{
		`
		func f9(a uint) int {
			return bits.LeadingZeros(a)
		}
		`,
		[]string{"\tCLZ\t"},
	},
}

var linuxPPC64LETests = []*asmTest{
	// Fused multiply-add/sub instructions.
	{
		`
		func f0(x, y, z float64) float64 {
			return x * y + z
		}
		`,
		[]string{"\tFMADD\t"},
	},
	{
		`
		func f1(x, y, z float64) float64 {
			return x * y - z
		}
		`,
		[]string{"\tFMSUB\t"},
	},
	{
		`
		func f2(x, y, z float32) float32 {
			return x * y + z
		}
		`,
		[]string{"\tFMADDS\t"},
	},
	{
		`
		func f3(x, y, z float32) float32 {
			return x * y - z
		}
		`,
		[]string{"\tFMSUBS\t"},
	},
	{
		`
		func f4(x uint32) uint32 {
			return x<<7 | x>>25
		}
		`,
		[]string{"\tROTLW\t"},
	},
	{
		`
		func f5(x uint32) uint32 {
			return x<<7 + x>>25
		}
		`,
		[]string{"\tROTLW\t"},
	},
	{
		`
		func f6(x uint32) uint32 {
			return x<<7 ^ x>>25
		}
		`,
		[]string{"\tROTLW\t"},
	},
	{
		`
		func f7(x uint64) uint64 {
			return x<<7 | x>>57
		}
		`,
		[]string{"\tROTL\t"},
	},
	{
		`
		func f8(x uint64) uint64 {
			return x<<7 + x>>57
		}
		`,
		[]string{"\tROTL\t"},
	},
	{
		`
		func f9(x uint64) uint64 {
			return x<<7 ^ x>>57
		}
		`,
		[]string{"\tROTL\t"},
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
