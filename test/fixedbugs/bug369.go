// +build !nacl,!windows
// run

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that compiling with optimization turned on produces faster code.

package main

import (
	"fmt"
	"go/build"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	a, err := build.ArchChar(build.Default.GOARCH)
	check(err)

	err = os.Chdir(filepath.Join(".", "fixedbugs", "bug369.dir"))
	check(err)

	run("go", "tool", a+"g", "-N", "-o", "slow."+a, "pkg.go")
	run("go", "tool", a+"g", "-o", "fast."+a, "pkg.go")
	run("go", "tool", a+"g", "-o", "main."+a, "main.go")
	run("go", "tool", a+"l", "-o", "a.exe", "main."+a)
	run("." + string(filepath.Separator) + "a.exe")

	os.Remove("slow." + a)
	os.Remove("fast." + a)
	os.Remove("main." + a)
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
