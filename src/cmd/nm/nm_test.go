// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

var testData uint32

func checkSymbols(t *testing.T, nmoutput []byte) {
	var checkSymbolsFound, testDataFound bool
	scanner := bufio.NewScanner(bytes.NewBuffer(nmoutput))
	for scanner.Scan() {
		f := strings.Fields(scanner.Text())
		if len(f) < 3 {
			continue
		}
		switch f[2] {
		case "cmd/nm.checkSymbols":
			checkSymbolsFound = true
			addr := "0x" + f[0]
			if addr != fmt.Sprintf("%p", checkSymbols) {
				t.Errorf("nm shows wrong address %v for checkSymbols (%p)", addr, checkSymbols)
			}
		case "cmd/nm.testData":
			testDataFound = true
			addr := "0x" + f[0]
			if addr != fmt.Sprintf("%p", &testData) {
				t.Errorf("nm shows wrong address %v for testData (%p)", addr, &testData)
			}
		}
	}
	if err := scanner.Err(); err != nil {
		t.Errorf("error while reading symbols: %v", err)
		return
	}
	if !checkSymbolsFound {
		t.Error("nm shows no checkSymbols symbol")
	}
	if !testDataFound {
		t.Error("nm shows no testData symbol")
	}
}

func TestNM(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	tmpDir, err := ioutil.TempDir("", "TestNM")
	if err != nil {
		t.Fatal("TempDir failed: ", err)
	}
	defer os.RemoveAll(tmpDir)

	testnmpath := filepath.Join(tmpDir, "testnm.exe")
	out, err := exec.Command("go", "build", "-o", testnmpath, "cmd/nm").CombinedOutput()
	if err != nil {
		t.Fatalf("go build -o %v cmd/nm: %v\n%s", testnmpath, err, string(out))
	}

	testfiles := []string{
		"elf/testdata/gcc-386-freebsd-exec",
		"elf/testdata/gcc-amd64-linux-exec",
		"macho/testdata/gcc-386-darwin-exec",
		"macho/testdata/gcc-amd64-darwin-exec",
		// "pe/testdata/gcc-amd64-mingw-exec", // no symbols!
		"pe/testdata/gcc-386-mingw-exec",
		"plan9obj/testdata/amd64-plan9-exec",
		"plan9obj/testdata/386-plan9-exec",
	}
	for _, f := range testfiles {
		exepath := filepath.Join(runtime.GOROOT(), "src", "debug", f)
		cmd := exec.Command(testnmpath, exepath)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Errorf("go tool nm %v: %v\n%s", exepath, err, string(out))
		}
	}

	cmd := exec.Command(testnmpath, os.Args[0])
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go tool nm %v: %v\n%s", os.Args[0], err, string(out))
	}
	checkSymbols(t, out)
}
