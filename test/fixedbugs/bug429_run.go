// +build !nacl
// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Run the bug429.go test.

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	cmd := exec.Command("go", "run", filepath.Join("fixedbugs", "bug429.go"))
	out, err := cmd.CombinedOutput()
	if err == nil {
		fmt.Println("expected deadlock")
		os.Exit(1)
	}

	want := "fatal error: all goroutines are asleep - deadlock!"
	got := string(out)
	if !strings.Contains(got, want) {
		fmt.Printf("got:\n%q\nshould contain:\n%q\n", got, want)
		os.Exit(1)
	}
}
