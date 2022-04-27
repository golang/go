// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package exec_test

import (
	"fmt"
	"internal/testenv"
	"os"
	"os/user"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func init() {
	registerHelperCommand("pwd", cmdPwd)
	registerHelperCommand("sleep", cmdSleep)
}

func cmdPwd(...string) {
	pwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(pwd)
}

func cmdSleep(args ...string) {
	n, err := strconv.Atoi(args[0])
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	time.Sleep(time.Duration(n) * time.Second)
}

func TestCredentialNoSetGroups(t *testing.T) {
	if runtime.GOOS == "android" {
		maySkipHelperCommand("echo")
		t.Skip("unsupported on Android")
	}

	u, err := user.Current()
	if err != nil {
		t.Fatalf("error getting current user: %v", err)
	}

	uid, err := strconv.Atoi(u.Uid)
	if err != nil {
		t.Fatalf("error converting Uid=%s to integer: %v", u.Uid, err)
	}

	gid, err := strconv.Atoi(u.Gid)
	if err != nil {
		t.Fatalf("error converting Gid=%s to integer: %v", u.Gid, err)
	}

	// If NoSetGroups is true, setgroups isn't called and cmd.Run should succeed
	cmd := helperCommand(t, "echo", "foo")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Credential: &syscall.Credential{
			Uid:         uint32(uid),
			Gid:         uint32(gid),
			NoSetGroups: true,
		},
	}

	if err = cmd.Run(); err != nil {
		t.Errorf("Failed to run command: %v", err)
	}
}

// For issue #19314: make sure that SIGSTOP does not cause the process
// to appear done.
func TestWaitid(t *testing.T) {
	t.Parallel()

	cmd := helperCommand(t, "sleep", "3")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}

	// The sleeps here are unnecessary in the sense that the test
	// should still pass, but they are useful to make it more
	// likely that we are testing the expected state of the child.
	time.Sleep(100 * time.Millisecond)

	if err := cmd.Process.Signal(syscall.SIGSTOP); err != nil {
		cmd.Process.Kill()
		t.Fatal(err)
	}

	ch := make(chan error)
	go func() {
		ch <- cmd.Wait()
	}()

	time.Sleep(100 * time.Millisecond)

	if err := cmd.Process.Signal(syscall.SIGCONT); err != nil {
		t.Error(err)
		syscall.Kill(cmd.Process.Pid, syscall.SIGCONT)
	}

	cmd.Process.Kill()

	<-ch
}

// https://go.dev/issue/50599: if Env is not set explicitly, setting Dir should
// implicitly update PWD to the correct path, and Environ should list the
// updated value.
func TestImplicitPWD(t *testing.T) {
	t.Parallel()

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name string
		dir  string
		want string
	}{
		{"empty", "", cwd},
		{"dot", ".", cwd},
		{"dotdot", "..", filepath.Dir(cwd)},
		{"PWD", cwd, cwd},
		{"PWDdotdot", cwd + string(filepath.Separator) + "..", filepath.Dir(cwd)},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cmd := helperCommand(t, "pwd")
			if cmd.Env != nil {
				t.Fatalf("test requires helperCommand not to set Env field")
			}
			cmd.Dir = tc.dir

			var pwds []string
			for _, kv := range cmd.Environ() {
				if strings.HasPrefix(kv, "PWD=") {
					pwds = append(pwds, strings.TrimPrefix(kv, "PWD="))
				}
			}

			wantPWDs := []string{tc.want}
			if tc.dir == "" {
				if _, ok := os.LookupEnv("PWD"); !ok {
					wantPWDs = nil
				}
			}
			if !reflect.DeepEqual(pwds, wantPWDs) {
				t.Errorf("PWD entries in cmd.Environ():\n\t%s\nwant:\n\t%s", strings.Join(pwds, "\n\t"), strings.Join(wantPWDs, "\n\t"))
			}

			cmd.Stderr = new(strings.Builder)
			out, err := cmd.Output()
			if err != nil {
				t.Fatalf("%v:\n%s", err, cmd.Stderr)
			}
			got := strings.Trim(string(out), "\r\n")
			t.Logf("in\n\t%s\n`pwd` reported\n\t%s", tc.dir, got)
			if got != tc.want {
				t.Errorf("want\n\t%s", tc.want)
			}
		})
	}
}

// However, if cmd.Env is set explicitly, setting Dir should not override it.
// (This checks that the implementation for https://go.dev/issue/50599 doesn't
// break existing users who may have explicitly mismatched the PWD variable.)
func TestExplicitPWD(t *testing.T) {
	maySkipHelperCommand("pwd")
	testenv.MustHaveSymlink(t)

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}

	link := filepath.Join(t.TempDir(), "link")
	if err := os.Symlink(cwd, link); err != nil {
		t.Fatal(err)
	}

	// Now link is another equally-valid name for cwd. If we set Dir to one and
	// PWD to the other, the subprocess should report the PWD version.
	cases := []struct {
		name string
		dir  string
		pwd  string
	}{
		{name: "original PWD", pwd: cwd},
		{name: "link PWD", pwd: link},
		{name: "in link with original PWD", dir: link, pwd: cwd},
		{name: "in dir with link PWD", dir: cwd, pwd: link},
		// Ideally we would also like to test what happens if we set PWD to
		// something totally bogus (or the empty string), but then we would have no
		// idea what output the subprocess should actually produce: cwd itself may
		// contain symlinks preserved from the PWD value in the test's environment.
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cmd := helperCommand(t, "pwd")
			// This is intentionally opposite to the usual order of setting cmd.Dir
			// and then calling cmd.Environ. Here, we *want* PWD not to match cmd.Dir,
			// so we don't care whether cmd.Dir is reflected in cmd.Environ.
			cmd.Env = append(cmd.Environ(), "PWD="+tc.pwd)
			cmd.Dir = tc.dir

			var pwds []string
			for _, kv := range cmd.Environ() {
				if strings.HasPrefix(kv, "PWD=") {
					pwds = append(pwds, strings.TrimPrefix(kv, "PWD="))
				}
			}

			wantPWDs := []string{tc.pwd}
			if !reflect.DeepEqual(pwds, wantPWDs) {
				t.Errorf("PWD entries in cmd.Environ():\n\t%s\nwant:\n\t%s", strings.Join(pwds, "\n\t"), strings.Join(wantPWDs, "\n\t"))
			}

			cmd.Stderr = new(strings.Builder)
			out, err := cmd.Output()
			if err != nil {
				t.Fatalf("%v:\n%s", err, cmd.Stderr)
			}
			got := strings.Trim(string(out), "\r\n")
			t.Logf("in\n\t%s\nwith PWD=%s\nsubprocess os.Getwd() reported\n\t%s", tc.dir, tc.pwd, got)
			if got != tc.pwd {
				t.Errorf("want\n\t%s", tc.pwd)
			}
		})
	}
}
