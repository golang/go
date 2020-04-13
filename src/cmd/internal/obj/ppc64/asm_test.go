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
	"strings"
	"testing"
)

var invalidPCAlignSrc = `
TEXT test(SB),0,$0-0
ADD $2, R3
PCALIGN $32
RET
`
var validPCAlignSrc = `
TEXT test(SB),0,$0-0
ADD $2, R3
PCALIGN $16
MOVD $8, R4
ADD $8, R4
PCALIGN $16
ADD $8, R4
PCALIGN $8
ADD $4, R6
PCALIGN $16
ADD R2, R3, R4
RET
`

// TestPCalign generates two asm files containing the
// PCALIGN directive, to verify correct values are and
// accepted, and incorrect values are flagged in error.
func TestPCalign(t *testing.T) {
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
