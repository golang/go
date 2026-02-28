// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package runtime_test

import (
	"bytes"
	"context"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func privesc(command string, args ...string) error {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
	defer cancel()
	var cmd *exec.Cmd
	if runtime.GOOS == "darwin" {
		cmd = exec.CommandContext(ctx, "sudo", append([]string{"-n", command}, args...)...)
	} else if runtime.GOOS == "openbsd" {
		cmd = exec.CommandContext(ctx, "doas", append([]string{"-n", command}, args...)...)
	} else {
		cmd = exec.CommandContext(ctx, "su", highPrivUser, "-c", fmt.Sprintf("%s %s", command, strings.Join(args, " ")))
	}
	_, err := cmd.CombinedOutput()
	return err
}

const highPrivUser = "root"

func setSetuid(t *testing.T, user, bin string) {
	t.Helper()
	// We escalate privileges here even if we are root, because for some reason on some builders
	// (at least freebsd-amd64-13_0) the default PATH doesn't include /usr/sbin, which is where
	// chown lives, but using 'su root -c' gives us the correct PATH.

	// buildTestProg uses os.MkdirTemp which creates directories with 0700, which prevents
	// setuid binaries from executing because of the missing g+rx, so we need to set the parent
	// directory to better permissions before anything else. We created this directory, so we
	// shouldn't need to do any privilege trickery.
	if err := privesc("chmod", "0777", filepath.Dir(bin)); err != nil {
		t.Skipf("unable to set permissions on %q, likely no passwordless sudo/su: %s", filepath.Dir(bin), err)
	}

	if err := privesc("chown", user, bin); err != nil {
		t.Skipf("unable to set permissions on test binary, likely no passwordless sudo/su: %s", err)
	}
	if err := privesc("chmod", "u+s", bin); err != nil {
		t.Skipf("unable to set permissions on test binary, likely no passwordless sudo/su: %s", err)
	}
}

func TestSUID(t *testing.T) {
	// This test is relatively simple, we build a test program which opens a
	// file passed via the TEST_OUTPUT envvar, prints the value of the
	// GOTRACEBACK envvar to stdout, and prints "hello" to stderr. We then chown
	// the program to "nobody" and set u+s on it. We execute the program, only
	// passing it two files, for stdin and stdout, and passing
	// GOTRACEBACK=system in the env.
	//
	// We expect that the program will trigger the SUID protections, resetting
	// the value of GOTRACEBACK, and opening the missing stderr descriptor, such
	// that the program prints "GOTRACEBACK=none" to stdout, and nothing gets
	// written to the file pointed at by TEST_OUTPUT.

	if *flagQuick {
		t.Skip("-quick")
	}

	testenv.MustHaveGoBuild(t)

	helloBin, err := buildTestProg(t, "testsuid")
	if err != nil {
		t.Fatal(err)
	}

	f, err := os.CreateTemp(t.TempDir(), "suid-output")
	if err != nil {
		t.Fatal(err)
	}
	tempfilePath := f.Name()
	f.Close()

	lowPrivUser := "nobody"
	setSetuid(t, lowPrivUser, helloBin)

	b := bytes.NewBuffer(nil)
	pr, pw, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}

	proc, err := os.StartProcess(helloBin, []string{helloBin}, &os.ProcAttr{
		Env:   []string{"GOTRACEBACK=system", "TEST_OUTPUT=" + tempfilePath},
		Files: []*os.File{os.Stdin, pw},
	})
	if err != nil {
		if os.IsPermission(err) {
			t.Skip("don't have execute permission on setuid binary, possibly directory permission issue?")
		}
		t.Fatal(err)
	}
	done := make(chan bool, 1)
	go func() {
		io.Copy(b, pr)
		pr.Close()
		done <- true
	}()
	ps, err := proc.Wait()
	if err != nil {
		t.Fatal(err)
	}
	pw.Close()
	<-done
	output := b.String()

	if ps.ExitCode() == 99 {
		t.Skip("binary wasn't setuid (uid == euid), unable to effectively test")
	}

	expected := "GOTRACEBACK=none\n"
	if output != expected {
		t.Errorf("unexpected output, got: %q, want %q", output, expected)
	}

	fc, err := os.ReadFile(tempfilePath)
	if err != nil {
		t.Fatal(err)
	}
	if string(fc) != "" {
		t.Errorf("unexpected file content, got: %q", string(fc))
	}

	// TODO: check the registers aren't leaked?
}
