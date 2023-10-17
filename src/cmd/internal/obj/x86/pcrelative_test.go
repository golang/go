// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86_test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"testing"
)

const asmData = `
GLOBL zeros<>(SB),8,$64
TEXT ·testASM(SB),4,$0
VMOVUPS zeros<>(SB), %s // PC relative relocation is off by 1, for Y8-Y15, Z8-15 and Z24-Z31
RET
`

const goData = `
package main

func testASM()

func main() {
	testASM()
}
`

func objdumpOutput(t *testing.T, mname, source string) []byte {
	tmpdir, err := os.MkdirTemp("", mname)
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	err = os.WriteFile(filepath.Join(tmpdir, "go.mod"), []byte(fmt.Sprintf("module %s\n", mname)), 0666)
	if err != nil {
		t.Fatal(err)
	}
	tmpfile, err := os.Create(filepath.Join(tmpdir, "input.s"))
	if err != nil {
		t.Fatal(err)
	}
	defer tmpfile.Close()
	_, err = tmpfile.WriteString(source)
	if err != nil {
		t.Fatal(err)
	}
	tmpfile2, err := os.Create(filepath.Join(tmpdir, "input.go"))
	if err != nil {
		t.Fatal(err)
	}
	defer tmpfile2.Close()
	_, err = tmpfile2.WriteString(goData)
	if err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t,
		testenv.GoToolPath(t), "build", "-o",
		filepath.Join(tmpdir, "output"))

	cmd.Env = append(os.Environ(),
		"GOARCH=amd64", "GOOS=linux", "GOPATH="+filepath.Join(tmpdir, "_gopath"))
	cmd.Dir = tmpdir

	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("error %s output %s", err, out)
	}
	cmd2 := testenv.Command(t,
		testenv.GoToolPath(t), "tool", "objdump", "-s", "testASM",
		filepath.Join(tmpdir, "output"))
	cmd2.Env = cmd.Env
	cmd2.Dir = tmpdir
	objout, err := cmd2.CombinedOutput()
	if err != nil {
		t.Fatalf("error %s output %s", err, objout)
	}

	return objout
}

func TestVexEvexPCrelative(t *testing.T) {
	testenv.MustHaveGoBuild(t)
LOOP:
	for _, reg := range []string{"Y0", "Y8", "Z0", "Z8", "Z16", "Z24"} {
		asm := fmt.Sprintf(asmData, reg)
		objout := objdumpOutput(t, "pcrelative", asm)
		data := bytes.Split(objout, []byte("\n"))
		for idx := len(data) - 1; idx >= 0; idx-- {
			// check that RET wasn't overwritten.
			if bytes.Index(data[idx], []byte("RET")) != -1 {
				if testing.Short() {
					break LOOP
				}
				continue LOOP
			}
		}
		t.Errorf("VMOVUPS zeros<>(SB), %s overwrote RET", reg)
	}
}
