// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

func TestNM(t *testing.T) {
	out, err := exec.Command("go", "build", "-o", "testnm.exe", "cmd/nm").CombinedOutput()
	if err != nil {
		t.Fatalf("go build -o testnm.exe cmd/nm: %v\n%s", err, string(out))
	}
	defer os.Remove("testnm.exe")

	testfiles := []string{
		"elf/testdata/gcc-386-freebsd-exec",
		"elf/testdata/gcc-amd64-linux-exec",
		"macho/testdata/gcc-386-darwin-exec",
		"macho/testdata/gcc-amd64-darwin-exec",
		"pe/testdata/gcc-amd64-mingw-exec",
		"pe/testdata/gcc-386-mingw-exec",
		"plan9obj/testdata/amd64-plan9-exec",
		"plan9obj/testdata/386-plan9-exec",
	}
	for _, f := range testfiles {
		exepath := filepath.Join(runtime.GOROOT(), "src", "pkg", "debug", f)
		cmd := exec.Command("./testnm.exe", exepath)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("go tool nm %v: %v\n%s", exepath, err, string(out))
		}
	}
}
