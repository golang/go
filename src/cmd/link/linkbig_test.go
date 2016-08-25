// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program generates a test to verify that a program can be
// successfully linked even when there are very large text
// sections present.

package main

import (
	"bytes"
	"cmd/internal/obj"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"testing"
)

func TestLargeText(t *testing.T) {

	var w bytes.Buffer

	if testing.Short() || (obj.GOARCH != "ppc64le" && obj.GOARCH != "ppc64") {
		t.Skip("Skipping large text section test in short mode or if not ppc64x")
	}
	const FN = 4
	tmpdir, err := ioutil.TempDir("", "bigtext")

	defer os.RemoveAll(tmpdir)

	// Generate the scenario where the total amount of text exceeds the
	// limit for the bl instruction, on RISC architectures like ppc64le,
	// which is 2^26.  When that happens the call requires special trampolines or
	// long branches inserted by the linker where supported.

	// Multiple .s files are generated instead of one.

	for j := 0; j < FN; j++ {
		testname := fmt.Sprintf("bigfn%d", j)
		fmt.Fprintf(&w, "TEXT ·%s(SB),$0\n", testname)
		for i := 0; i < 2200000; i++ {
			fmt.Fprintf(&w, "\tMOVD\tR0,R3\n")
		}
		fmt.Fprintf(&w, "\tRET\n")
		err := ioutil.WriteFile(tmpdir+"/"+testname+".s", w.Bytes(), 0666)
		if err != nil {
			t.Fatalf("can't write output: %v\n", err)
		}
		w.Reset()
	}
	fmt.Fprintf(&w, "package main\n")
	fmt.Fprintf(&w, "\nimport (\n")
	fmt.Fprintf(&w, "\t\"os\"\n")
	fmt.Fprintf(&w, "\t\"fmt\"\n")
	fmt.Fprintf(&w, ")\n\n")

	for i := 0; i < FN; i++ {
		fmt.Fprintf(&w, "func bigfn%d()\n", i)
	}
	fmt.Fprintf(&w, "\nfunc main() {\n")

	// There are lots of dummy code generated in the .s files just to generate a lot
	// of text. Link them in but guard their call so their code is not executed but
	// the main part of the program can be run.

	fmt.Fprintf(&w, "\tif os.Getenv(\"LINKTESTARG\") != \"\" {\n")
	for i := 0; i < FN; i++ {
		fmt.Fprintf(&w, "\t\tbigfn%d()\n", i)
	}
	fmt.Fprintf(&w, "\t}\n")
	fmt.Fprintf(&w, "\tfmt.Printf(\"PASS\\n\")\n")
	fmt.Fprintf(&w, "}")
	err = ioutil.WriteFile(tmpdir+"/bigfn.go", w.Bytes(), 0666)
	if err != nil {
		t.Fatalf("can't write output: %v\n", err)
	}

	os.Chdir(tmpdir)
	cmd := exec.Command("go", "build", "-o", "bigtext", "-ldflags", "'-linkmode=external'")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Build of big text program failed: %v, output: %s", err, out)
	}
	cmd = exec.Command(tmpdir + "/bigtext")
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Program failed with err %v, output: %s", err, out)
	}
}
