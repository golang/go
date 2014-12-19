// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/build"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
)

func main() {
	if runtime.Compiler != "gc" || runtime.GOOS == "nacl" {
		return
	}
	a, err := build.ArchChar(runtime.GOARCH)
	if err != nil {
		fmt.Println("BUG:", err)
		os.Exit(1)
	}
	out := run("go", "tool", a+"g", "-S", filepath.Join("fixedbugs", "issue9355.dir", "a.go"))
	// 6g/8g print the offset as dec, but 5g/9g print the offset as hex.
	patterns := []string{
		`rel 0\+\d t=1 \"\"\.x\+8\r?\n`,       // y = &x.b
		`rel 0\+\d t=1 \"\"\.x\+(28|1c)\r?\n`, // z = &x.d.q
		`rel 0\+\d t=1 \"\"\.b\+5\r?\n`,       // c = &b[5]
		`rel 0\+\d t=1 \"\"\.x\+(88|58)\r?\n`, // w = &x.f[3].r
	}
	for _, p := range patterns {
		if ok, err := regexp.Match(p, out); !ok || err != nil {
			println(string(out))
			panic("can't find pattern " + p)
		}
	}
}

func run(cmd string, args ...string) []byte {
	out, err := exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
		fmt.Println(string(out))
		fmt.Println(err)
		os.Exit(1)
	}
	return out
}
