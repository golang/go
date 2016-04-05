// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"bytes"
	"internal/testenv"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Make sure "hello world" does not link in all the
// fmt.scanf routines. See issue 6853.
func TestScanfRemoval(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// Make a directory to work in.
	dir, err := ioutil.TempDir("", "issue6853a-")
	if err != nil {
		log.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Create source.
	src := filepath.Join(dir, "test.go")
	f, err := os.Create(src)
	if err != nil {
		log.Fatalf("could not create source file: %v", err)
	}
	f.Write([]byte(`
package main
import "fmt"
func main() {
	fmt.Println("hello world")
}
`))
	f.Close()

	// Name of destination.
	dst := filepath.Join(dir, "test")

	// Compile source.
	cmd := exec.Command("go", "build", "-o", dst, src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatalf("could not build target: %v", err)
	}

	// Check destination to see if scanf code was included.
	cmd = exec.Command("go", "tool", "nm", dst)
	out, err = cmd.CombinedOutput()
	if err != nil {
		log.Fatalf("could not read target: %v", err)
	}
	if bytes.Contains(out, []byte("scanInt")) {
		log.Fatalf("scanf code not removed from helloworld")
	}
}

// Make sure -S prints assembly code. See issue 14515.
func TestDashS(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// Make a directory to work in.
	dir, err := ioutil.TempDir("", "issue14515-")
	if err != nil {
		log.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Create source.
	src := filepath.Join(dir, "test.go")
	f, err := os.Create(src)
	if err != nil {
		log.Fatalf("could not create source file: %v", err)
	}
	f.Write([]byte(`
package main
import "fmt"
func main() {
	fmt.Println("hello world")
}
`))
	f.Close()

	// Compile source.
	cmd := exec.Command("go", "build", "-gcflags", "-S", "-o", filepath.Join(dir, "test"), src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatalf("could not build target: %v", err)
	}

	patterns := []string{
		// It is hard to look for actual instructions in an
		// arch-independent way. So we'll just look for
		// pseudo-ops that are arch-independent.
		"\tTEXT\t",
		"\tFUNCDATA\t",
		"\tPCDATA\t",
	}
	outstr := string(out)
	for _, p := range patterns {
		if !strings.Contains(outstr, p) {
			println(outstr)
			panic("can't find pattern " + p)
		}
	}
}
