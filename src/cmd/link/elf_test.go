// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build dragonfly freebsd linux netbsd openbsd

package main

import (
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

var asmSource = `
	.section .text1,"ax"
s1:
	.byte 0
	.section .text2,"ax"
s2:
	.byte 0
`

var goSource = `
package main
func main() {}
`

// The linker used to crash if an ELF input file had multiple text sections
// with the same name.
func TestSectionsWithSameName(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	t.Parallel()

	objcopy, err := exec.LookPath("objcopy")
	if err != nil {
		t.Skipf("can't find objcopy: %v", err)
	}

	dir, err := ioutil.TempDir("", "go-link-TestSectionsWithSameName")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	gopath := filepath.Join(dir, "GOPATH")
	env := append(os.Environ(), "GOPATH="+gopath)

	if err := ioutil.WriteFile(filepath.Join(dir, "go.mod"), []byte("module elf_test\n"), 0666); err != nil {
		t.Fatal(err)
	}

	asmFile := filepath.Join(dir, "x.s")
	if err := ioutil.WriteFile(asmFile, []byte(asmSource), 0444); err != nil {
		t.Fatal(err)
	}

	goTool := testenv.GoToolPath(t)
	cmd := exec.Command(goTool, "env", "CC")
	cmd.Env = env
	ccb, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	cc := strings.TrimSpace(string(ccb))

	cmd = exec.Command(goTool, "env", "GOGCCFLAGS")
	cmd.Env = env
	cflagsb, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	cflags := strings.Fields(string(cflagsb))

	asmObj := filepath.Join(dir, "x.o")
	t.Logf("%s %v -c -o %s %s", cc, cflags, asmObj, asmFile)
	if out, err := exec.Command(cc, append(cflags, "-c", "-o", asmObj, asmFile)...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	asm2Obj := filepath.Join(dir, "x2.syso")
	t.Logf("%s --rename-section .text2=.text1 %s %s", objcopy, asmObj, asm2Obj)
	if out, err := exec.Command(objcopy, "--rename-section", ".text2=.text1", asmObj, asm2Obj).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	for _, s := range []string{asmFile, asmObj} {
		if err := os.Remove(s); err != nil {
			t.Fatal(err)
		}
	}

	goFile := filepath.Join(dir, "main.go")
	if err := ioutil.WriteFile(goFile, []byte(goSource), 0444); err != nil {
		t.Fatal(err)
	}

	cmd = exec.Command(goTool, "build")
	cmd.Dir = dir
	cmd.Env = env
	t.Logf("%s build", goTool)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}
}
