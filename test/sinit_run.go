// +build !nacl,!js,gc
// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Run the sinit test.

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
)

func main() {
	f, err := ioutil.TempFile("", "sinit-*.o")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	f.Close()

	cmd := exec.Command("go", "tool", "compile", "-p=sinit", "-o", f.Name(), "-S", "sinit.go")
	out, err := cmd.CombinedOutput()
	os.Remove(f.Name())
	if err != nil {
		fmt.Println(string(out))
		fmt.Println(err)
		os.Exit(1)
	}

	if len(bytes.TrimSpace(out)) == 0 {
		fmt.Println("'go tool compile -S sinit.go' printed no output")
		os.Exit(1)
	}
	if bytes.Contains(out, []byte("initdone")) {
		fmt.Println("sinit generated an init function")
		os.Exit(1)
	}
}
