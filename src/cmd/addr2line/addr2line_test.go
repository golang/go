// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"bytes"
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestMain executes the test binary as the addr2line command if
// GO_ADDR2LINETEST_IS_ADDR2LINE is set, and runs the tests otherwise.
func TestMain(m *testing.M) {
	if os.Getenv("GO_ADDR2LINETEST_IS_ADDR2LINE") != "" {
		main()
		os.Exit(0)
	}

	os.Setenv("GO_ADDR2LINETEST_IS_ADDR2LINE", "1") // Set for subprocesses to inherit.
	os.Exit(m.Run())
}

func loadSyms(t *testing.T, dbgExePath string) map[string]string {
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "nm", dbgExePath)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %v\n%s", cmd, err, string(out))
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

func runAddr2Line(t *testing.T, dbgExePath, addr string) (funcname, path, lineno string) {
	cmd := testenv.Command(t, testenv.Executable(t), dbgExePath)
	cmd.Stdin = strings.NewReader(addr)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go tool addr2line %v: %v\n%s", os.Args[0], err, string(out))
	}
	f := strings.Split(string(out), "\n")
	if len(f) < 3 && f[2] == "" {
		t.Fatal("addr2line output must have 2 lines")
	}
	funcname = f[0]
	pathAndLineNo := f[1]
	f = strings.Split(pathAndLineNo, ":")
	if runtime.GOOS == "windows" && len(f) == 3 {
		// Reattach drive letter.
		f = []string{f[0] + ":" + f[1], f[2]}
	}
	if len(f) != 2 {
		t.Fatalf("no line number found in %q", pathAndLineNo)
	}
	return funcname, f[0], f[1]
}

const symName = "cmd/addr2line.TestAddr2Line"

func testAddr2Line(t *testing.T, dbgExePath, addr string) {
	funcName, srcPath, srcLineNo := runAddr2Line(t, dbgExePath, addr)
	if symName != funcName {
		t.Fatalf("expected function name %v; got %v", symName, funcName)
	}
	fi1, err := os.Stat("addr2line_test.go")
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}

	// Debug paths are stored slash-separated, so convert to system-native.
	srcPath = filepath.FromSlash(srcPath)
	fi2, err := os.Stat(srcPath)
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if !os.SameFile(fi1, fi2) {
		t.Fatalf("addr2line_test.go and %s are not same file", srcPath)
	}
	if want := "102"; srcLineNo != want {
		t.Fatalf("line number = %v; want %s", srcLineNo, want)
	}
}

// This is line 101. The test depends on that.
func TestAddr2Line(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	tmpDir := t.TempDir()

	// Build copy of test binary with debug symbols,
	// since the one running now may not have them.
	exepath := filepath.Join(tmpDir, "testaddr2line_test.exe")
	out, err := testenv.Command(t, testenv.GoToolPath(t), "test", "-c", "-o", exepath, "cmd/addr2line").CombinedOutput()
	if err != nil {
		t.Fatalf("go test -c -o %v cmd/addr2line: %v\n%s", exepath, err, string(out))
	}

	syms := loadSyms(t, exepath)

	testAddr2Line(t, exepath, syms[symName])
	testAddr2Line(t, exepath, "0x"+syms[symName])
}
