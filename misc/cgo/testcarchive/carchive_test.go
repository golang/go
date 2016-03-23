// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package carchive_test

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"unicode"
)

// Program to run.
var bin []string

// C compiler wiht args (from $(go env CC) $(go env GOGCCFLAGS)).
var cc []string

// An environment with GOPATH=$(pwd).
var gopathEnv []string

// ".exe" on Windows.
var exeSuffix string

func init() {
	bin = []string{"./testp"}
	execScript := "go_" + runtime.GOOS + "_" + runtime.GOARCH + "_exec"
	if executor, err := exec.LookPath(execScript); err == nil {
		bin = []string{executor, "./testp"}
	}

	out, err := exec.Command("go", "env", "CC").Output()
	if err != nil {
		fmt.Fprintf(os.Stderr, "go env CC failed:\n%s", err)
		fmt.Fprintf(os.Stderr, "%s", err.(*exec.ExitError).Stderr)
		os.Exit(2)
	}
	cc = []string{strings.TrimSpace(string(out))}

	out, err = exec.Command("go", "env", "GOGCCFLAGS").Output()
	if err != nil {
		fmt.Fprintf(os.Stderr, "go env GOGCCFLAGS failed:\n%s", err)
		fmt.Fprintf(os.Stderr, "%s", err.(*exec.ExitError).Stderr)
		os.Exit(2)
	}
	quote := '\000'
	start := 0
	lastSpace := true
	backslash := false
	s := string(out)
	for i, c := range s {
		if quote == '\000' && unicode.IsSpace(c) {
			if !lastSpace {
				cc = append(cc, s[start:i])
				lastSpace = true
			}
		} else {
			if lastSpace {
				start = i
				lastSpace = false
			}
			if quote == '\000' && !backslash && (c == '"' || c == '\'') {
				quote = c
				backslash = false
			} else if !backslash && quote == c {
				quote = '\000'
			} else if (quote == '\000' || quote == '"') && !backslash && c == '\\' {
				backslash = true
			} else {
				backslash = false
			}
		}
	}
	if !lastSpace {
		cc = append(cc, s[start:])
	}

	if runtime.GOOS == "darwin" {
		cc = append(cc, "-Wl,-no_pie")

		// For Darwin/ARM.
		// TODO(crawshaw): can we do better?
		cc = append(cc, []string{"-framework", "CoreFoundation", "-framework", "Foundation"}...)
	}
	cc = append(cc, "-I", filepath.Join("pkg", runtime.GOOS+"_"+runtime.GOARCH))

	// Build an environment with GOPATH=$(pwd)
	env := os.Environ()
	var n []string
	for _, e := range env {
		if !strings.HasPrefix(e, "GOPATH=") {
			n = append(n, e)
		}
	}
	dir, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	n = append(n, "GOPATH="+dir)
	gopathEnv = n

	if runtime.GOOS == "windows" {
		exeSuffix = ".exe"
	}
}

func TestInstall(t *testing.T) {
	defer func() {
		os.Remove("libgo.a")
		os.Remove("libgo.h")
		os.Remove("testp")
		os.RemoveAll("pkg")
	}()

	cmd := exec.Command("go", "install", "-buildmode=c-archive", "libgo")
	cmd.Env = gopathEnv
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	ccArgs := append(cc, "-o", "testp"+exeSuffix, "main.c", filepath.Join("pkg", runtime.GOOS+"_"+runtime.GOARCH, "libgo.a"))
	if out, err := exec.Command(ccArgs[0], ccArgs[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	binArgs := append(bin, "arg1", "arg2")
	if out, err := exec.Command(binArgs[0], binArgs[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	os.Remove("libgo.a")
	os.Remove("libgo.h")
	os.Remove("testp")

	// Test building libgo other than installing it.
	// Header files are now present.
	cmd = exec.Command("go", "build", "-buildmode=c-archive", filepath.Join("src", "libgo", "libgo.go"))
	cmd.Env = gopathEnv
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	ccArgs = append(cc, "-o", "testp"+exeSuffix, "main.c", "libgo.a")
	if out, err := exec.Command(ccArgs[0], ccArgs[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	if out, err := exec.Command(binArgs[0], binArgs[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	os.Remove("libgo.a")
	os.Remove("libgo.h")
	os.Remove("testp")

	cmd = exec.Command("go", "build", "-buildmode=c-archive", "-o", "libgo.a", "libgo")
	cmd.Env = gopathEnv
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	if out, err := exec.Command(ccArgs[0], ccArgs[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	if out, err := exec.Command(binArgs[0], binArgs[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}
}

func TestEarlySignalHandler(t *testing.T) {
	switch runtime.GOOS {
	case "darwin":
		switch runtime.GOARCH {
		case "arm", "arm64":
			t.Skipf("skipping on %s/%s; see https://golang.org/issue/13701", runtime.GOOS, runtime.GOARCH)
		}
	case "windows":
		t.Skip("skipping signal test on Windows")
	}

	defer func() {
		os.Remove("libgo2.a")
		os.Remove("libgo2.h")
		os.Remove("testp")
		os.RemoveAll("pkg")
	}()

	cmd := exec.Command("go", "build", "-buildmode=c-archive", "-o", "libgo2.a", "libgo2")
	cmd.Env = gopathEnv
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	ccArgs := append(cc, "-o", "testp"+exeSuffix, "main2.c", "libgo2.a")
	if out, err := exec.Command(ccArgs[0], ccArgs[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	if out, err := exec.Command(bin[0], bin[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}
}

func TestSignalForwarding(t *testing.T) {
	switch runtime.GOOS {
	case "darwin":
		switch runtime.GOARCH {
		case "arm", "arm64":
			t.Skipf("skipping on %s/%s; see https://golang.org/issue/13701", runtime.GOOS, runtime.GOARCH)
		}
	case "windows":
		t.Skip("skipping signal test on Windows")
	}

	defer func() {
		os.Remove("libgo2.a")
		os.Remove("libgo2.h")
		os.Remove("testp")
		os.RemoveAll("pkg")
	}()

	cmd := exec.Command("go", "build", "-buildmode=c-archive", "-o", "libgo2.a", "libgo2")
	cmd.Env = gopathEnv
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	ccArgs := append(cc, "-o", "testp"+exeSuffix, "main5.c", "libgo2.a")
	if out, err := exec.Command(ccArgs[0], ccArgs[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	cmd = exec.Command(bin[0], append(bin[1:], "1")...)

	out, err := cmd.CombinedOutput()

	if err == nil {
		t.Logf("%s", out)
		t.Error("test program succeeded unexpectedly")
	} else if ee, ok := err.(*exec.ExitError); !ok {
		t.Logf("%s", out)
		t.Errorf("error (%v) has type %T; expected exec.ExitError", err, err)
	} else if ws, ok := ee.Sys().(syscall.WaitStatus); !ok {
		t.Logf("%s", out)
		t.Errorf("error.Sys (%v) has type %T; expected syscall.WaitStatus", ee.Sys(), ee.Sys())
	} else if !ws.Signaled() || ws.Signal() != syscall.SIGSEGV {
		t.Logf("%s", out)
		t.Errorf("got %v; expected SIGSEGV", ee)
	}
}

func TestSignalForwardingExternal(t *testing.T) {
	switch runtime.GOOS {
	case "darwin":
		switch runtime.GOARCH {
		case "arm", "arm64":
			t.Skipf("skipping on %s/%s; see https://golang.org/issue/13701", runtime.GOOS, runtime.GOARCH)
		}
	case "windows":
		t.Skip("skipping signal test on Windows")
	}

	defer func() {
		os.Remove("libgo2.a")
		os.Remove("libgo2.h")
		os.Remove("testp")
		os.RemoveAll("pkg")
	}()

	cmd := exec.Command("go", "build", "-buildmode=c-archive", "-o", "libgo2.a", "libgo2")
	cmd.Env = gopathEnv
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	ccArgs := append(cc, "-o", "testp"+exeSuffix, "main5.c", "libgo2.a")
	if out, err := exec.Command(ccArgs[0], ccArgs[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	cmd = exec.Command(bin[0], append(bin[1:], "2")...)

	stderr, err := cmd.StderrPipe()
	if err != nil {
		t.Fatal(err)
	}
	defer stderr.Close()

	r := bufio.NewReader(stderr)

	err = cmd.Start()

	if err != nil {
		t.Fatal(err)
	}

	// Wait for trigger to ensure that the process is started.
	ok, err := r.ReadString('\n')

	// Verify trigger.
	if err != nil || ok != "OK\n" {
		t.Fatalf("Did not receive OK signal")
	}

	// Trigger an interrupt external to the process.
	cmd.Process.Signal(syscall.SIGSEGV)

	err = cmd.Wait()

	if err == nil {
		t.Error("test program succeeded unexpectedly")
	} else if ee, ok := err.(*exec.ExitError); !ok {
		t.Errorf("error (%v) has type %T; expected exec.ExitError", err, err)
	} else if ws, ok := ee.Sys().(syscall.WaitStatus); !ok {
		t.Errorf("error.Sys (%v) has type %T; expected syscall.WaitStatus", ee.Sys(), ee.Sys())
	} else if !ws.Signaled() || ws.Signal() != syscall.SIGSEGV {
		t.Errorf("got %v; expected SIGSEGV", ee)
	}
}

func TestOsSignal(t *testing.T) {
	switch runtime.GOOS {
	case "windows":
		t.Skip("skipping signal test on Windows")
	}

	defer func() {
		os.Remove("libgo3.a")
		os.Remove("libgo3.h")
		os.Remove("testp")
		os.RemoveAll("pkg")
	}()

	cmd := exec.Command("go", "build", "-buildmode=c-archive", "-o", "libgo3.a", "libgo3")
	cmd.Env = gopathEnv
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	ccArgs := append(cc, "-o", "testp"+exeSuffix, "main3.c", "libgo3.a")
	if out, err := exec.Command(ccArgs[0], ccArgs[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	if out, err := exec.Command(bin[0], bin[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}
}

func TestSigaltstack(t *testing.T) {
	switch runtime.GOOS {
	case "windows":
		t.Skip("skipping signal test on Windows")
	}

	defer func() {
		os.Remove("libgo4.a")
		os.Remove("libgo4.h")
		os.Remove("testp")
		os.RemoveAll("pkg")
	}()

	cmd := exec.Command("go", "build", "-buildmode=c-archive", "-o", "libgo4.a", "libgo4")
	cmd.Env = gopathEnv
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	ccArgs := append(cc, "-o", "testp"+exeSuffix, "main4.c", "libgo4.a")
	if out, err := exec.Command(ccArgs[0], ccArgs[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	if out, err := exec.Command(bin[0], bin[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}
}

const testar = `#!/usr/bin/env bash
while expr $1 : '[-]' >/dev/null; do
  shift
done
echo "testar" > $1
echo "testar" > PWD/testar.ran
`

func TestExtar(t *testing.T) {
	switch runtime.GOOS {
	case "windows":
		t.Skip("skipping signal test on Windows")
	}

	defer func() {
		os.Remove("libgo4.a")
		os.Remove("libgo4.h")
		os.Remove("testar")
		os.Remove("testar.ran")
		os.RemoveAll("pkg")
	}()

	os.Remove("testar")
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	s := strings.Replace(testar, "PWD", dir, 1)
	if err := ioutil.WriteFile("testar", []byte(s), 0777); err != nil {
		t.Fatal(err)
	}

	cmd := exec.Command("go", "build", "-buildmode=c-archive", "-ldflags=-extar="+filepath.Join(dir, "testar"), "-o", "libgo4.a", "libgo4")
	cmd.Env = gopathEnv
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}

	if _, err := os.Stat("testar.ran"); err != nil {
		if os.IsNotExist(err) {
			t.Error("testar does not exist after go build")
		} else {
			t.Errorf("error checking testar: %v", err)
		}
	}
}
