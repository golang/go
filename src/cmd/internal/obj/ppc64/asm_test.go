// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64

import (
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
