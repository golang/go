// run

//go:build !nacl && !js && !wasip1 && !gccgo

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we don't get an index out of bounds error
// while trying to print a map that is concurrently modified.
// The runtime might complain (throw) if it detects the modification,
// so we have to run the test as a subprocess.

package main

import (
	"os/exec"
	"strings"
)

func main() {
	out, _ := exec.Command("go", "run", "fixedbugs/issue33275.go").CombinedOutput()
	if strings.Contains(string(out), "index out of range") {
		panic(`go run issue33275.go reported "index out of range"`)
	}
}
