// Copyright 2016 The Go Authors. All rights reserved.
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
	"strings"
	"syscall"
	"testing"
	"time"
	"unicode"
)

// Program to run.
var bin []string

// C compiler with args (from $(go env CC) $(go env GOGCCFLAGS)).
var cc []string

// An environment with GOPATH=$(pwd).
var gopathEnv []string

// ".exe" on Windows.
var exeSuffix string

var GOOS, GOARCH string
var libgodir string

func init() {
	bin = []string{"./testp"}
	GOOS = goEnv("GOOS")
	GOARCH = goEnv("GOARCH")
	execScript := "go_" + GOOS + "_" + GOARCH + "_exec"
	if executor, err := exec.LookPath(execScript); err == nil {
		bin = []string{executor, "./testp"}
	}

	ccOut := goEnv("CC")
	cc = []string{string(ccOut)}

	out := goEnv("GOGCCFLAGS")
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

	if GOOS == "darwin" {
		// For Darwin/ARM.
		// TODO(crawshaw): can we do better?
		cc = append(cc, []string{"-framework", "CoreFoundation", "-framework", "Foundation"}...)
	}
	libgodir = GOOS + "_" + GOARCH
	if GOOS == "darwin" && (GOARCH == "arm" || GOARCH == "arm64") {
		libgodir = GOOS + "_" + GOARCH + "_shared"
	}
	cc = append(cc, "-I", filepath.Join("pkg", libgodir))

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

	if GOOS == "windows" {
		exeSuffix = ".exe"
	}
}

func goEnv(key string) string {
	out, err := exec.Command("go", "env", key).Output()
	if err != nil {
		fmt.Fprintf(os.Stderr, "go env %s failed:\n%s", key, err)
		fmt.Fprintf(os.Stderr, "%s", err.(*exec.ExitError).Stderr)
		os.Exit(2)
	}
	return strings.TrimSpace(string(out))
}

func compilemain(t *testing.T, libgo string) {
	ccArgs := append(cc, "-o", "testp"+exeSuffix, "main.c")
	if GOOS == "windows" {
		ccArgs = append(ccArgs, "main_windows.c", libgo, "-lntdll", "-lws2_32", "-lwinmm")
	} else {
		ccArgs = append(ccArgs, "main_unix.c", libgo)
	}
	t.Log(ccArgs)

	if out, err := exec.Command(ccArgs[0], ccArgs[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
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

	compilemain(t, filepath.Join("pkg", libgodir, "libgo.a"))

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

	compilemain(t, "libgo.a")

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

	compilemain(t, "libgo.a")

	if out, err := exec.Command(binArgs[0], binArgs[1:]...).CombinedOutput(); err != nil {
		t.Logf("%s", out)
		t.Fatal(err)
	}
}

func TestEarlySignalHandler(t *testing.T) {
	switch GOOS {
	case "darwin":
		switch GOARCH {
		case "arm", "arm64":
			t.Skipf("skipping on %s/%s; see https://golang.org/issue/13701", GOOS, GOARCH)
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
	switch GOOS {
	case "darwin":
		switch GOARCH {
		case "arm", "arm64":
			t.Skipf("skipping on %s/%s; see https://golang.org/issue/13701", GOOS, GOARCH)
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
	switch GOOS {
	case "darwin":
		switch GOARCH {
		case "arm", "arm64":
			t.Skipf("skipping on %s/%s; see https://golang.org/issue/13701", GOOS, GOARCH)
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

	// We want to send the process a signal and see if it dies.
	// Normally the signal goes to the C thread, the Go signal
	// handler picks it up, sees that it is running in a C thread,
	// and the program dies. Unfortunately, occasionally the
	// signal is delivered to a Go thread, which winds up
	// discarding it because it was sent by another program and
	// there is no Go handler for it. To avoid this, run the
	// program several times in the hopes that it will eventually
	// fail.
	const tries = 20
	for i := 0; i < tries; i++ {
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

		// Give the program a chance to enter the sleep function.
		time.Sleep(time.Millisecond)

		cmd.Process.Signal(syscall.SIGSEGV)

		err = cmd.Wait()

		if err == nil {
			continue
		}

		if ee, ok := err.(*exec.ExitError); !ok {
			t.Errorf("error (%v) has type %T; expected exec.ExitError", err, err)
		} else if ws, ok := ee.Sys().(syscall.WaitStatus); !ok {
			t.Errorf("error.Sys (%v) has type %T; expected syscall.WaitStatus", ee.Sys(), ee.Sys())
		} else if !ws.Signaled() || ws.Signal() != syscall.SIGSEGV {
			t.Errorf("got %v; expected SIGSEGV", ee)
		} else {
			// We got the error we expected.
			return
		}
	}

	t.Errorf("program succeeded unexpectedly %d times", tries)
}

func TestOsSignal(t *testing.T) {
	switch GOOS {
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
	switch GOOS {
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
	switch GOOS {
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
