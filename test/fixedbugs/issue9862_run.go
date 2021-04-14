// +build !nacl,!js,gc
// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check for compile or link error.

package main

import (
	"os/exec"
	"strings"
)

func main() {
	out, err := exec.Command("go", "run", "fixedbugs/issue9862.go").CombinedOutput()
	outstr := string(out)
	if err == nil {
		println("go run issue9862.go succeeded, should have failed\n", outstr)
		return
	}
	if !strings.Contains(outstr, "symbol too large") {
		println("go run issue9862.go gave unexpected error; want symbol too large:\n", outstr)
	}
}
