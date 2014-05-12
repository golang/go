// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func loadSyms(t *testing.T) map[string]string {
	cmd := exec.Command("go", "tool", "nm", os.Args[0])
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go tool nm %v: %v\n%s", os.Args[0], err, string(out))
	}
	syms := make(map[string]string)
	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		f := strings.Fields(scanner.Text())
		if len(f) < 3 {
			continue
		}
		syms[f[2]] = f[0]
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("error reading symbols: %v", err)
	}
	return syms
}

func runObjDump(t *testing.T, exepath, startaddr string) (path, lineno string) {
	addr, err := strconv.ParseUint(startaddr, 16, 64)
	if err != nil {
		t.Fatalf("invalid start address %v: %v", startaddr, err)
	}
	endaddr := fmt.Sprintf("%x", addr+10)
	cmd := exec.Command(exepath, os.Args[0], "0x"+startaddr, "0x"+endaddr)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go tool objdump %v: %v\n%s", os.Args[0], err, string(out))
	}
	f := strings.Split(string(out), "\n")
	if len(f) < 1 {
		t.Fatal("objdump output must have at least one line")
	}
	pathAndLineNo := f[0]
	f = strings.Split(pathAndLineNo, ":")
	if runtime.GOOS == "windows" {
		switch len(f) {
		case 2:
			return f[0], f[1]
		case 3:
			return f[0] + ":" + f[1], f[2]
		default:
			t.Fatalf("no line number found in %q", pathAndLineNo)
		}
	}
	if len(f) != 2 {
		t.Fatalf("no line number found in %q", pathAndLineNo)
	}
	return f[0], f[1]
}

// This is line 75.  The test depends on that.
func TestObjDump(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skip("skipping test; see http://golang.org/issue/7947")
	}
	syms := loadSyms(t)

	tmpDir, err := ioutil.TempDir("", "TestObjDump")
	if err != nil {
		t.Fatal("TempDir failed: ", err)
	}
	defer os.RemoveAll(tmpDir)

	exepath := filepath.Join(tmpDir, "testobjdump.exe")
	out, err := exec.Command("go", "build", "-o", exepath, "cmd/objdump").CombinedOutput()
	if err != nil {
		t.Fatalf("go build -o %v cmd/objdump: %v\n%s", exepath, err, string(out))
	}

	srcPath, srcLineNo := runObjDump(t, exepath, syms["cmd/objdump.TestObjDump"])
	fi1, err := os.Stat("objdump_test.go")
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	fi2, err := os.Stat(srcPath)
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if !os.SameFile(fi1, fi2) {
		t.Fatalf("objdump_test.go and %s are not same file", srcPath)
	}
	if srcLineNo != "76" {
		t.Fatalf("line number = %v; want 76", srcLineNo)
	}
}
