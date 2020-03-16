// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
)

func main() {
	if runtime.Compiler != "gc" || runtime.GOOS == "nacl" || runtime.GOOS == "js" {
		return
	}

	err := os.Chdir(filepath.Join("fixedbugs", "issue9355.dir"))
	check(err)

	f, err := ioutil.TempFile("", "issue9355-*.o")
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	f.Close()

	out := run("go", "tool", "compile", "-o", f.Name(), "-S", "a.go")
	os.Remove(f.Name())

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

func check(err error) {
	if err != nil {
		fmt.Println("BUG:", err)
		os.Exit(1)
	}
}
