// +build !nacl,!js,!windows
// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that compiling with optimization turned on produces faster code.

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	err := os.Chdir(filepath.Join(".", "fixedbugs", "bug369.dir"))
	check(err)

	tmpDir, err := ioutil.TempDir("", "bug369")
	check(err)
	defer os.RemoveAll(tmpDir)

	tmp := func(name string) string {
		return filepath.Join(tmpDir, name)
	}

	run("go", "tool", "compile", "-N", "-o", tmp("slow.o"), "pkg.go")
	run("go", "tool", "compile", "-o", tmp("fast.o"), "pkg.go")
	run("go", "tool", "compile", "-D", tmpDir, "-o", tmp("main.o"), "main.go")
	run("go", "tool", "link", "-o", tmp("a.exe"), tmp("main.o"))
	run(tmp("a.exe"))
}

func run(name string, args ...string) {
	cmd := exec.Command(name, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(string(out))
		fmt.Println(err)
		os.Exit(1)
	}
}

func check(err error) {
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
