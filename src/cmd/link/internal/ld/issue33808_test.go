// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"internal/testenv"
	"io/ioutil"
	"os"
	"runtime"
	"strings"
	"testing"
)

const prog = `
package main

import "log"

func main() {
	log.Fatalf("HERE")
}
`

func TestIssue33808(t *testing.T) {
	if runtime.GOOS != "darwin" {
		return
	}
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)

	dir, err := ioutil.TempDir("", "TestIssue33808")
	if err != nil {
		t.Fatalf("could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	f := gobuild(t, dir, prog, "-ldflags=-linkmode=external")
	f.Close()

	syms, err := f.Symbols()
	if err != nil {
		t.Fatalf("Error reading symbols: %v", err)
	}

	name := "log.Fatalf"
	for _, sym := range syms {
		if strings.Contains(sym.Name, name) {
			return
		}
	}
	t.Fatalf("Didn't find %v", name)
}
