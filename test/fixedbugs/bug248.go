// +build !nacl,!plan9,!windows
// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
)

func main() {
	// TODO: If we get rid of errchk, re-enable this test on Windows.
	errchk, err := filepath.Abs("errchk")
	check(err)

	bugDir := filepath.Join(".", "fixedbugs", "bug248.dir")
	run("go", "tool", "compile", filepath.Join(bugDir, "bug0.go"))
	run("go", "tool", "compile", filepath.Join(bugDir, "bug1.go"))
	run("go", "tool", "compile", filepath.Join(bugDir, "bug2.go"))
	run(errchk, "go", "tool", "compile", "-e", filepath.Join(bugDir, "bug3.go"))
	run("go", "tool", "link", "bug2.o")
	run(fmt.Sprintf(".%ca.out", filepath.Separator))

	os.Remove("bug0.o")
	os.Remove("bug1.o")
	os.Remove("bug2.o")
	os.Remove("a.out")
}

var bugRE = regexp.MustCompile(`(?m)^BUG`)

func run(name string, args ...string) {
	cmd := exec.Command(name, args...)
	out, err := cmd.CombinedOutput()
	if bugRE.Match(out) || err != nil {
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
