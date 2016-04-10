// +build !nacl,!windows
// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that compiling with optimization turned on produces faster code.

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	err := os.Chdir(filepath.Join(".", "fixedbugs", "bug369.dir"))
	check(err)

	run("go", "tool", "compile", "-N", "-o", "slow.o", "pkg.go")
	run("go", "tool", "compile", "-o", "fast.o", "pkg.go")
	run("go", "tool", "compile", "-o", "main.o", "main.go")
	run("go", "tool", "link", "-o", "a.exe", "main.o")
	run("." + string(filepath.Separator) + "a.exe")

	os.Remove("slow.o")
	os.Remove("fast.o")
	os.Remove("main.o")
	os.Remove("a.exe")
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
