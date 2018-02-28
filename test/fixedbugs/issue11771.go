// +build !nacl
// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 11771: Magic comments should ignore carriage returns.

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

func main() {
	if runtime.Compiler != "gc" {
		return
	}

	dir, err := ioutil.TempDir("", "go-issue11771")
	if err != nil {
		log.Fatalf("creating temp dir: %v\n", err)
	}
	defer os.RemoveAll(dir)

	// The go:nowritebarrier magic comment is only permitted in
	// the runtime package.  So we confirm that the compilation
	// fails.

	var buf bytes.Buffer
	fmt.Fprintln(&buf, `
package main

func main() {
}
`)
	fmt.Fprintln(&buf, "//go:nowritebarrier\r")
	fmt.Fprintln(&buf, `
func x() {
}
`)

	if err := ioutil.WriteFile(filepath.Join(dir, "x.go"), buf.Bytes(), 0666); err != nil {
		log.Fatal(err)
	}

	cmd := exec.Command("go", "tool", "compile", "x.go")
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	if err == nil {
		log.Fatal("compile succeeded unexpectedly")
	}
	if !bytes.Contains(output, []byte("only allowed in runtime")) {
		log.Fatalf("wrong error message from compiler; got:\n%s\n", output)
	}
}
