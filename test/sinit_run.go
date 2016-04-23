// +build !nacl
// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Run the sinit test.

package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
)

func main() {
	cmd := exec.Command("go", "tool", "compile", "-S", "sinit.go")
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(string(out))
		fmt.Println(err)
		os.Exit(1)
	}
	os.Remove("sinit.o")

	if bytes.Contains(out, []byte("initdone")) {
		fmt.Println("sinit generated an init function")
		os.Exit(1)
	}
}
