// +build !nacl,!plan9,!windows
// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

	// TODO: If we get rid of errchk, re-enable this test on Windows.
	errchk, err := filepath.Abs("errchk")
	check(err)

	err = os.Chdir(filepath.Join("fixedbugs", "bug248.dir"))
	check(err)

	run("go", "tool", a+"g", "bug0.go")
	run("go", "tool", a+"g", "bug1.go")
	run("go", "tool", a+"g", "bug2.go")
	run(errchk, "go", "tool", a+"g", "-e", "bug3.go")
	run("go", "tool", a+"l", "bug2."+a)
	run(fmt.Sprintf(".%c%s.out", filepath.Separator, a))

	os.Remove("bug0." + a)
	os.Remove("bug1." + a)
	os.Remove("bug2." + a)
	os.Remove(a + ".out")
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
