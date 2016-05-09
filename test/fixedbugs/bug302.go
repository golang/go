// +build !nacl
// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	run("go", "tool", "compile", filepath.Join("fixedbugs", "bug302.dir", "p.go"))
	run("go", "tool", "pack", "grc", "pp.a", "p.o")
	run("go", "tool", "compile", "-I", ".", filepath.Join("fixedbugs", "bug302.dir", "main.go"))
	os.Remove("p.o")
	os.Remove("pp.a")
	os.Remove("main.o")
}

func run(cmd string, args ...string) {
	out, err := exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
		fmt.Println(string(out))
		fmt.Println(err)
		os.Exit(1)
	}
}
