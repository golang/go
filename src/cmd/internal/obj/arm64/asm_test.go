// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"testing"
)

// TestLarge generates a very large file to verify that large
// program builds successfully, in particular, too-far
// conditional branches are fixed.
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

	// generate a very large function
	buf := bytes.NewBuffer(make([]byte, 0, 7000000))
	gen(buf)

	tmpfile := filepath.Join(dir, "x.s")
	err = ioutil.WriteFile(tmpfile, buf.Bytes(), 0644)
	if err != nil {
		t.Fatalf("can't write output: %v\n", err)
	}

	// build generated file
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=arm64", "GOOS=linux")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Build failed: %v, output: %s", err, out)
	}
}

// gen generates a very large program, with a very far conditional branch.
func gen(buf *bytes.Buffer) {
	fmt.Fprintln(buf, "TEXT f(SB),0,$0-0")
	fmt.Fprintln(buf, "TBZ $5, R0, label")
	fmt.Fprintln(buf, "CBZ R0, label")
	fmt.Fprintln(buf, "BEQ label")
	for i := 0; i < 1<<19; i++ {
		fmt.Fprintln(buf, "MOVD R0, R1")
	}
	fmt.Fprintln(buf, "label:")
	fmt.Fprintln(buf, "RET")
}

// Issue 20348.
func TestNoRet(t *testing.T) {
	dir, err := ioutil.TempDir("", "testnoret")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	tmpfile := filepath.Join(dir, "x.s")
	if err := ioutil.WriteFile(tmpfile, []byte("TEXT ·stub(SB),$0-0\nNOP\n"), 0644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=arm64", "GOOS=linux")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Errorf("%v\n%s", err, out)
	}
}

// TestPCALIGN verifies the correctness of the PCALIGN by checking if the
// code can be aligned to the alignment value.
func TestPCALIGN(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir, err := ioutil.TempDir("", "testpcalign")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	tmpfile := filepath.Join(dir, "test.s")

	code1 := []byte("TEXT ·foo(SB),$0-0\nMOVD $0, R0\nPCALIGN $8\nMOVD $1, R1\nRET\n")
	code2 := []byte("TEXT ·foo(SB),$0-0\nMOVD $0, R0\nPCALIGN $16\nMOVD $2, R2\nRET\n")
	// If the output contains this pattern, the pc-offsite of "MOVD $1, R1" is 8 bytes aligned.
	out1 := `0x0008\s00008\s\(.*\)\tMOVD\t\$1,\sR1`
	// If the output contains this pattern, the pc-offsite of "MOVD $2, R2" is 16 bytes aligned.
	out2 := `0x0010\s00016\s\(.*\)\tMOVD\t\$2,\sR2`
	var testCases = []struct {
		name string
		code []byte
		out  string
	}{
		{"8-byte alignment", code1, out1},
		{"16-byte alignment", code2, out2},
	}

	for _, test := range testCases {
		if err := ioutil.WriteFile(tmpfile, test.code, 0644); err != nil {
			t.Fatal(err)
		}
		cmd := exec.Command(testenv.GoToolPath(t), "tool", "asm", "-S", tmpfile)
		cmd.Env = append(os.Environ(), "GOARCH=arm64", "GOOS=linux")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("The %s build failed: %v, output: %s", test.name, err, out)
			continue
		}

		matched, err := regexp.MatchString(test.out, string(out))
		if err != nil {
			t.Fatal(err)
		}
		if !matched {
			t.Errorf("The %s testing failed!\ninput: %s\noutput: %s\n", test.name, test.code, out)
		}
	}
}
