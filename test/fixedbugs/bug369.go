// run

//go:build !nacl && !js && !wasip1 && gc

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that compiling with optimization turned on produces faster code.

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	err := os.Chdir(filepath.Join(".", "fixedbugs", "bug369.dir"))
	check(err)

	tmpDir, err := ioutil.TempDir("", "bug369")
	check(err)
	defer os.RemoveAll(tmpDir)

	tmp := func(name string) string {
		return filepath.Join(tmpDir, name)
	}

	check(os.Mkdir(tmp("test"), 0777))

	stdlibimportcfg, err := os.ReadFile(os.Getenv("STDLIB_IMPORTCFG"))
	check(err)
	importcfg := string(stdlibimportcfg) + "\npackagefile test/slow=" + tmp("test/slow.o") + "\npackagefile test/fast=" + tmp("test/fast.o")
	os.WriteFile(tmp("importcfg"), []byte(importcfg), 0644)

	run("go", "tool", "compile", "-importcfg="+tmp("importcfg"), "-p=test/slow", "-N", "-o", tmp("test/slow.o"), "pkg.go")
	run("go", "tool", "compile", "-importcfg="+tmp("importcfg"), "-p=test/fast", "-o", tmp("test/fast.o"), "pkg.go")
	run("go", "tool", "compile", "-importcfg="+tmp("importcfg"), "-p=main", "-D", "test", "-o", tmp("main.o"), "main.go")
	run("go", "tool", "link", "-importcfg="+tmp("importcfg"), "-o", tmp("a.exe"), tmp("main.o"))
	run(tmp("a.exe"))
}

func run(name string, args ...string) {
	cmd := exec.Command(name, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Println(string(out))
		fmt.Println(err)
		os.Exit(1)
	}
}

func check(err error) {
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
