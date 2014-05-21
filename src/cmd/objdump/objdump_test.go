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
	if runtime.GOOS == "nacl" {
		t.Skip("skipping on nacl")
	}

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

func runObjDump(t *testing.T, exe, startaddr, endaddr string) (path, lineno string) {
	if runtime.GOOS == "nacl" {
		t.Skip("skipping on nacl")
	}

	cmd := exec.Command(exe, os.Args[0], startaddr, endaddr)
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

func testObjDump(t *testing.T, exe, startaddr, endaddr string, line int) {
	srcPath, srcLineNo := runObjDump(t, exe, startaddr, endaddr)
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
	if srcLineNo != fmt.Sprint(line) {
		t.Fatalf("line number = %v; want %d", srcLineNo, line)
	}
}

func TestObjDump(t *testing.T) {
	_, _, line, _ := runtime.Caller(0)
	syms := loadSyms(t)

	tmp, exe := buildObjdump(t)
	defer os.RemoveAll(tmp)

	startaddr := syms["cmd/objdump.TestObjDump"]
	addr, err := strconv.ParseUint(startaddr, 16, 64)
	if err != nil {
		t.Fatalf("invalid start address %v: %v", startaddr, err)
	}
	endaddr := fmt.Sprintf("%x", addr+10)
	testObjDump(t, exe, startaddr, endaddr, line-1)
	testObjDump(t, exe, "0x"+startaddr, "0x"+endaddr, line-1)
}

func buildObjdump(t *testing.T) (tmp, exe string) {
	if runtime.GOOS == "nacl" {
		t.Skip("skipping on nacl")
	}

	tmp, err := ioutil.TempDir("", "TestObjDump")
	if err != nil {
		t.Fatal("TempDir failed: ", err)
	}

	exe = filepath.Join(tmp, "testobjdump.exe")
	out, err := exec.Command("go", "build", "-o", exe, "cmd/objdump").CombinedOutput()
	if err != nil {
		os.RemoveAll(tmp)
		t.Fatalf("go build -o %v cmd/objdump: %v\n%s", exe, err, string(out))
	}
	return
}

var x86Need = []string{
	"fmthello.go:6",
	"TEXT main.main(SB)",
	"JMP main.main(SB)",
	"CALL fmt.Println(SB)",
	"RET",
}

var armNeed = []string{
	"fmthello.go:6",
	"TEXT main.main(SB)",
	"B main.main(SB)",
	"BL fmt.Println(SB)",
	"RET",
}

// objdump is fully cross platform: it can handle binaries
// from any known operating system and architecture.
// We could in principle add binaries to testdata and check
// all the supported systems during this test. However, the
// binaries would be about 1 MB each, and we don't want to
// add that much junk to the hg repository. Instead, build a
// binary for the current system (only) and test that objdump
// can handle that one.

func TestDisasm(t *testing.T) {
	tmp, exe := buildObjdump(t)
	defer os.RemoveAll(tmp)

	hello := filepath.Join(tmp, "hello.exe")
	out, err := exec.Command("go", "build", "-o", hello, "testdata/fmthello.go").CombinedOutput()
	if err != nil {
		t.Fatalf("go build fmthello.go: %v\n%s", err, out)
	}
	need := []string{
		"fmthello.go:6",
		"TEXT main.main(SB)",
	}
	switch runtime.GOARCH {
	case "amd64", "386":
		need = append(need, x86Need...)
	case "arm":
		need = append(need, armNeed...)
		t.Skip("disassembler not ready on arm yet")
	}

	out, err = exec.Command(exe, "-s", "main.main", hello).CombinedOutput()
	if err != nil {
		t.Fatalf("objdump fmthello.exe: %v\n%s", err, out)
	}

	text := string(out)
	ok := true
	for _, s := range need {
		if !strings.Contains(text, s) {
			t.Errorf("disassembly missing '%s'", s)
			ok = false
		}
	}
	if !ok {
		t.Logf("full disassembly:\n%s", text)
	}
}
