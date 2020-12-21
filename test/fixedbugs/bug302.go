// +build !nacl,!js,gc
// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
)

var tmpDir string

func main() {
	fb, err := filepath.Abs("fixedbugs")
	if err == nil {
		tmpDir, err = ioutil.TempDir("", "bug302")
	}
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer os.RemoveAll(tmpDir)

	run("go", "tool", "compile", filepath.Join(fb, "bug302.dir", "p.go"))
	run("go", "tool", "pack", "grc", "pp.a", "p.o")
	run("go", "tool", "compile", "-I", ".", filepath.Join(fb, "bug302.dir", "main.go"))
}

func run(cmd string, args ...string) {
	c := exec.Command(cmd, args...)
	c.Dir = tmpDir
	out, err := c.CombinedOutput()
	if err != nil {
		fmt.Println(string(out))
		fmt.Println(err)
		os.Exit(1)
	}
}
