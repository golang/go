// +build !nacl,!plan9,!windows
// run

// Copyright 2011 The Go Authors. All rights reserved.
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
	// TODO: If we get rid of errchk, re-enable this test on Plan 9 and Windows.
	errchk, err := filepath.Abs("errchk")
	check(err)

	bugDir := filepath.Join(".", "fixedbugs", "bug345.dir")
	run("go", "tool", "compile", filepath.Join(bugDir, "io.go"))
	run(errchk, "go", "tool", "compile", "-e", filepath.Join(bugDir, "main.go"))

	os.Remove("io.o")
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
