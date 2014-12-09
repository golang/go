// +build !nacl
// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/build"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

func main() {
	a, err := build.ArchChar(runtime.GOARCH)
	if err != nil {
		fmt.Println("BUG:", err)
		os.Exit(1)
	}

	run("go", "tool", a+"g", filepath.Join("fixedbugs", "bug302.dir", "p.go"))
	run("go", "tool", "pack", "grc", "pp.a", "p."+a)
	run("go", "tool", a+"g", "-I", ".", filepath.Join("fixedbugs", "bug302.dir", "main.go"))
	os.Remove("p."+a)
	os.Remove("pp.a")
	os.Remove("main."+a)
}

func run(cmd string, args ...string) {
	out, err := exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
		fmt.Println(string(out))
		fmt.Println(err)
		os.Exit(1)
	}
}
