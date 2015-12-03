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
	"path"
	"testing"
)

// Make sure "hello world" does not link in all the
// fmt.scanf routines.  See issue 6853.
func TestScanfRemoval(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// Make a directory to work in.
	dir, err := ioutil.TempDir("", "issue6853a-")
	if err != nil {
		log.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Create source.
	src := path.Join(dir, "test.go")
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
	dst := path.Join(dir, "test")

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
	if bytes.Index(out, []byte("scanInt")) != -1 {
		log.Fatalf("scanf code not removed from helloworld")
	}
}
