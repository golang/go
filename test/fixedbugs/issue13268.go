// run

//go:build gc

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test error message when EOF is encountered in the
// middle of a BOM.
//
// Since the error requires an EOF, we cannot use the
// errorcheckoutput mechanism.

package main

import (
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"strings"
)

func main() {
	// create source
	f, err := ioutil.TempFile("", "issue13268-")
	if err != nil {
		log.Fatalf("could not create source file: %v", err)
	}
	f.Write([]byte("package p\n\nfunc \xef\xef")) // if this fails, we will die later
	f.Close()
	defer os.Remove(f.Name())

	// compile and test output
	cmd := exec.Command("go", "tool", "compile", f.Name())
	out, err := cmd.CombinedOutput()
	if err == nil {
		log.Fatalf("expected cmd/compile to fail")
	}
	if strings.HasPrefix(string(out), "illegal UTF-8 sequence") {
		log.Fatalf("error %q not found", out)
	}
}
