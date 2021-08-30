// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

var invalidPCAlignSrc = `
TEXT test(SB),0,$0-0
ADD $2, R3
PCALIGN $64
RET
`

var validPCAlignSrc = `
TEXT test(SB),0,$0-0
ADD $2, R3
PCALIGN $16
MOVD $8, R16
ADD $8, R4
PCALIGN $32
ADD $8, R3
PCALIGN $8
ADD $4, R8
RET
`

var platformEnvs = [][]string{
	{"GOOS=aix", "GOARCH=ppc64"},
	{"GOOS=linux", "GOARCH=ppc64"},
	{"GOOS=linux", "GOARCH=ppc64le"},
}

// TestLarge generates a very large file to verify that large
// program builds successfully, and branches which exceed the
// range of BC are rewritten to reach.
func TestLarge(t *testing.T) {
	if testing.Short() {
		t.Skip("Skip in short mode")
	}
	testenv.MustHaveGoBuild(t)

	dir, err := ioutil.TempDir("", "testlarge")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// A few interesting test cases for long conditional branch fixups
	tests := []struct {
		jmpinsn     string
		backpattern []string
		fwdpattern  []string
	}{
		// Test the interesting cases of conditional branch rewrites for too-far targets. Simple conditional
		// branches can be made to reach with one JMP insertion, compound conditionals require two.
		//
		// TODO: BI is interpreted as a register (the R???x/R0 should be $x)
		// beq <-> bne conversion (insert one jump)
		{"BEQ",
			[]string{``,
				`0x20030 131120\s\(.*\)\tBC\t\$4,\sR\?\?\?2,\s131128`,
				`0x20034 131124\s\(.*\)\tJMP\t0`},
			[]string{``,
				`0x0000 00000\s\(.*\)\tBC\t\$4,\sR\?\?\?2,\s8`,
				`0x0004 00004\s\(.*\)\tJMP\t131128`},
		},
		{"BNE",
			[]string{``,
				`0x20030 131120\s\(.*\)\tBC\t\$12,\sR\?\?\?2,\s131128`,
				`0x20034 131124\s\(.*\)\tJMP\t0`},
			[]string{``,
				`0x0000 00000\s\(.*\)\tBC\t\$12,\sR\?\?\?2,\s8`,
				`0x0004 00004\s\(.*\)\tJMP\t131128`}},
		// bdnz (BC 16,0,tgt) <-> bdz (BC 18,0,+4) conversion (insert one jump)
		{"BC 16,0,",
			[]string{``,
				`0x20030 131120\s\(.*\)\tBC\t\$18,\s131128`,
				`0x20034 131124\s\(.*\)\tJMP\t0`},
			[]string{``,
				`0x0000 00000\s\(.*\)\tBC\t\$18,\s8`,
				`0x0004 00004\s\(.*\)\tJMP\t131128`}},
		{"BC 18,0,",
			[]string{``,
				`0x20030 131120\s\(.*\)\tBC\t\$16,\s131128`,
				`0x20034 131124\s\(.*\)\tJMP\t0`},
			[]string{``,
				`0x0000 00000\s\(.*\)\tBC\t\$16,\s8`,
				`0x0004 00004\s\(.*\)\tJMP\t131128`}},
		// bdnzt (BC 8,0,tgt) <-> bdnzt (BC 8,0,+4) conversion (insert two jumps)
		{"BC 8,0,",
			[]string{``,
				`0x20034 131124\s\(.*\)\tBC\t\$8,\sR0,\s131132`,
				`0x20038 131128\s\(.*\)\tJMP\t131136`,
				`0x2003c 131132\s\(.*\)\tJMP\t0\n`},
			[]string{``,
				`0x0000 00000\s\(.*\)\tBC\t\$8,\sR0,\s8`,
				`0x0004 00004\s\(.*\)\tJMP\t12`,
				`0x0008 00008\s\(.*\)\tJMP\t131136\n`}},
	}

	for _, test := range tests {
		// generate a very large function
		buf := bytes.NewBuffer(make([]byte, 0, 7000000))
		gen(buf, test.jmpinsn)

		tmpfile := filepath.Join(dir, "x.s")
		err = ioutil.WriteFile(tmpfile, buf.Bytes(), 0644)
		if err != nil {
			t.Fatalf("can't write output: %v\n", err)
		}

		// Test on all supported ppc64 platforms
		for _, platenv := range platformEnvs {
			cmd := exec.Command(testenv.GoToolPath(t), "tool", "asm", "-S", "-o", filepath.Join(dir, "test.o"), tmpfile)
			cmd.Env = append(os.Environ(), platenv...)
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Errorf("Assemble failed (%v): %v, output: %s", platenv, err, out)
			}
			matched, err := regexp.MatchString(strings.Join(test.fwdpattern, "\n\t*"), string(out))
			if err != nil {
				t.Fatal(err)
			}
			if !matched {
				t.Errorf("Failed to detect long foward BC fixup in (%v):%s\n", platenv, out)
			}
			matched, err = regexp.MatchString(strings.Join(test.backpattern, "\n\t*"), string(out))
			if err != nil {
				t.Fatal(err)
			}
			if !matched {
				t.Errorf("Failed to detect long backward BC fixup in (%v):%s\n", platenv, out)
			}
		}
	}
}

// gen generates a very large program with a very long forward and backwards conditional branch.
func gen(buf *bytes.Buffer, jmpinsn string) {
	fmt.Fprintln(buf, "TEXT f(SB),0,$0-0")
	fmt.Fprintln(buf, "label_start:")
	fmt.Fprintln(buf, jmpinsn, "label_end")
	for i := 0; i < (1<<15 + 10); i++ {
		fmt.Fprintln(buf, "MOVD R0, R1")
	}
	fmt.Fprintln(buf, jmpinsn, "label_start")
	fmt.Fprintln(buf, "label_end:")
	fmt.Fprintln(buf, "MOVD R0, R1")
	fmt.Fprintln(buf, "RET")
}

// TestPCalign generates two asm files containing the
// PCALIGN directive, to verify correct values are and
// accepted, and incorrect values are flagged in error.
func TestPCalign(t *testing.T) {
	var pattern8 = `0x...8\s.*ADD\s..,\sR8`
	var pattern16 = `0x...[80]\s.*MOVD\s..,\sR16`
	var pattern32 = `0x...0\s.*ADD\s..,\sR3`

	testenv.MustHaveGoBuild(t)

	dir, err := ioutil.TempDir("", "testpcalign")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// generate a test with valid uses of PCALIGN

	tmpfile := filepath.Join(dir, "x.s")
	err = ioutil.WriteFile(tmpfile, []byte(validPCAlignSrc), 0644)
	if err != nil {
		t.Fatalf("can't write output: %v\n", err)
	}

	// build generated file without errors and assemble it
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), "-S", tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=ppc64le", "GOOS=linux")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Build failed: %v, output: %s", err, out)
	}

	matched, err := regexp.MatchString(pattern8, string(out))
	if err != nil {
		t.Fatal(err)
	}
	if !matched {
		t.Errorf("The 8 byte alignment is not correct: %t, output:%s\n", matched, out)
	}

	matched, err = regexp.MatchString(pattern16, string(out))
	if err != nil {
		t.Fatal(err)
	}
	if !matched {
		t.Errorf("The 16 byte alignment is not correct: %t, output:%s\n", matched, out)
	}

	matched, err = regexp.MatchString(pattern32, string(out))
	if err != nil {
		t.Fatal(err)
	}
	if !matched {
		t.Errorf("The 32 byte alignment is not correct: %t, output:%s\n", matched, out)
	}

	// generate a test with invalid use of PCALIGN

	tmpfile = filepath.Join(dir, "xi.s")
	err = ioutil.WriteFile(tmpfile, []byte(invalidPCAlignSrc), 0644)
	if err != nil {
		t.Fatalf("can't write output: %v\n", err)
	}

	// build test with errors and check for messages
	cmd = exec.Command(testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "xi.o"), "-S", tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=ppc64le", "GOOS=linux")
	out, err = cmd.CombinedOutput()
	if !strings.Contains(string(out), "Unexpected alignment") {
		t.Errorf("Invalid alignment not detected for PCALIGN\n")
	}
}

// Verify register constants are correctly aligned. Much of the ppc64 assembler assumes masking out significant
// bits will produce a valid register number:
// REG_Rx & 31 == x
// REG_Fx & 31 == x
// REG_Vx & 31 == x
// REG_VSx & 63 == x
// REG_SPRx & 1023 == x
// REG_CRx & 7 == x
//
// VR and FPR disjointly overlap VSR, interpreting as VSR registers should produce the correctly overlapped VSR.
// REG_FPx & 63 == x
// REG_Vx & 63 == x + 32
func TestRegValueAlignment(t *testing.T) {
	tstFunc := func(rstart, rend, msk, rout int) {
		for i := rstart; i <= rend; i++ {
			if i&msk != rout {
				t.Errorf("%v is not aligned to 0x%X (expected %d, got %d)\n", rconv(i), msk, rout, rstart&msk)
			}
			rout++
		}
	}
	var testType = []struct {
		rstart int
		rend   int
		msk    int
		rout   int
	}{
		{REG_VS0, REG_VS63, 63, 0},
		{REG_R0, REG_R31, 31, 0},
		{REG_F0, REG_F31, 31, 0},
		{REG_V0, REG_V31, 31, 0},
		{REG_V0, REG_V31, 63, 32},
		{REG_F0, REG_F31, 63, 0},
		{REG_SPR0, REG_SPR0 + 1023, 1023, 0},
		{REG_CR0, REG_CR7, 7, 0},
	}
	for _, t := range testType {
		tstFunc(t.rstart, t.rend, t.msk, t.rout)
	}
}
