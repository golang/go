// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && cgo
// +build linux,cgo

package cgotest

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"syscall"
	"testing"
)

// #include <stdio.h>
// #include <stdlib.h>
// #include <pthread.h>
// #include <unistd.h>
// #include <sys/types.h>
//
// pthread_t *t = NULL;
// pthread_mutex_t mu;
// int nts = 0;
// int all_done = 0;
//
// static void *aFn(void *vargp) {
//   int done = 0;
//   while (!done) {
//     usleep(100);
//     pthread_mutex_lock(&mu);
//     done = all_done;
//     pthread_mutex_unlock(&mu);
//   }
//   return NULL;
// }
//
// void trial(int argc) {
//   int i;
//   nts = argc;
//   t = calloc(nts, sizeof(pthread_t));
//   pthread_mutex_init(&mu, NULL);
//   for (i = 0; i < nts; i++) {
//     pthread_create(&t[i], NULL, aFn, NULL);
//   }
// }
//
// void cleanup(void) {
//   int i;
//   pthread_mutex_lock(&mu);
//   all_done = 1;
//   pthread_mutex_unlock(&mu);
//   for (i = 0; i < nts; i++) {
//     pthread_join(t[i], NULL);
//   }
//   pthread_mutex_destroy(&mu);
//   free(t);
// }
import "C"

// compareStatus is used to confirm the contents of the thread
// specific status files match expectations.
func compareStatus(filter, expect string) error {
	expected := filter + expect
	pid := syscall.Getpid()
	fs, err := os.ReadDir(fmt.Sprintf("/proc/%d/task", pid))
	if err != nil {
		return fmt.Errorf("unable to find %d tasks: %v", pid, err)
	}
	expectedProc := fmt.Sprintf("Pid:\t%d", pid)
	foundAThread := false
	for _, f := range fs {
		tf := fmt.Sprintf("/proc/%s/status", f.Name())
		d, err := os.ReadFile(tf)
		if err != nil {
			// There are a surprising number of ways this
			// can error out on linux.  We've seen all of
			// the following, so treat any error here as
			// equivalent to the "process is gone":
			//    os.IsNotExist(err),
			//    "... : no such process",
			//    "... : bad file descriptor.
			continue
		}
		lines := strings.Split(string(d), "\n")
		for _, line := range lines {
			// Different kernel vintages pad differently.
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "Pid:\t") {
				// On loaded systems, it is possible
				// for a TID to be reused really
				// quickly. As such, we need to
				// validate that the thread status
				// info we just read is a task of the
				// same process PID as we are
				// currently running, and not a
				// recently terminated thread
				// resurfaced in a different process.
				if line != expectedProc {
					break
				}
				// Fall through in the unlikely case
				// that filter at some point is
				// "Pid:\t".
			}
			if strings.HasPrefix(line, filter) {
				if line == expected {
					foundAThread = true
					break
				}
				if filter == "Groups:" && strings.HasPrefix(line, "Groups:\t") {
					// https://github.com/golang/go/issues/46145
					// Containers don't reliably output this line in sorted order so manually sort and compare that.
					a := strings.Split(line[8:], " ")
					sort.Strings(a)
					got := strings.Join(a, " ")
					if got == expected[8:] {
						foundAThread = true
						break
					}

				}
				return fmt.Errorf("%q got:%q want:%q (bad) [pid=%d file:'%s' %v]\n", tf, line, expected, pid, string(d), expectedProc)
			}
		}
	}
	if !foundAThread {
		return fmt.Errorf("found no thread /proc/<TID>/status files for process %q", expectedProc)
	}
	return nil
}

// test1435 test 9 glibc implemented setuid/gid syscall functions are
// mapped.  This test is a slightly more expansive test than that of
// src/syscall/syscall_linux_test.go:TestSetuidEtc() insofar as it
// launches concurrent threads from C code via CGo and validates that
// they are subject to the system calls being tested. For the actual
// Go functionality being tested here, the syscall_linux_test version
// is considered authoritative, but non-trivial improvements to that
// should be mirrored here.
func test1435(t *testing.T) {
	if syscall.Getuid() != 0 {
		t.Skip("skipping root only test")
	}

	// Launch some threads in C.
	const cts = 5
	C.trial(cts)
	defer C.cleanup()

	vs := []struct {
		call           string
		fn             func() error
		filter, expect string
	}{
		{call: "Setegid(1)", fn: func() error { return syscall.Setegid(1) }, filter: "Gid:", expect: "\t0\t1\t0\t1"},
		{call: "Setegid(0)", fn: func() error { return syscall.Setegid(0) }, filter: "Gid:", expect: "\t0\t0\t0\t0"},

		{call: "Seteuid(1)", fn: func() error { return syscall.Seteuid(1) }, filter: "Uid:", expect: "\t0\t1\t0\t1"},
		{call: "Setuid(0)", fn: func() error { return syscall.Setuid(0) }, filter: "Uid:", expect: "\t0\t0\t0\t0"},

		{call: "Setgid(1)", fn: func() error { return syscall.Setgid(1) }, filter: "Gid:", expect: "\t1\t1\t1\t1"},
		{call: "Setgid(0)", fn: func() error { return syscall.Setgid(0) }, filter: "Gid:", expect: "\t0\t0\t0\t0"},

		{call: "Setgroups([]int{0,1,2,3})", fn: func() error { return syscall.Setgroups([]int{0, 1, 2, 3}) }, filter: "Groups:", expect: "\t0 1 2 3"},
		{call: "Setgroups(nil)", fn: func() error { return syscall.Setgroups(nil) }, filter: "Groups:", expect: ""},
		{call: "Setgroups([]int{0})", fn: func() error { return syscall.Setgroups([]int{0}) }, filter: "Groups:", expect: "\t0"},

		{call: "Setregid(101,0)", fn: func() error { return syscall.Setregid(101, 0) }, filter: "Gid:", expect: "\t101\t0\t0\t0"},
		{call: "Setregid(0,102)", fn: func() error { return syscall.Setregid(0, 102) }, filter: "Gid:", expect: "\t0\t102\t102\t102"},
		{call: "Setregid(0,0)", fn: func() error { return syscall.Setregid(0, 0) }, filter: "Gid:", expect: "\t0\t0\t0\t0"},

		{call: "Setreuid(1,0)", fn: func() error { return syscall.Setreuid(1, 0) }, filter: "Uid:", expect: "\t1\t0\t0\t0"},
		{call: "Setreuid(0,2)", fn: func() error { return syscall.Setreuid(0, 2) }, filter: "Uid:", expect: "\t0\t2\t2\t2"},
		{call: "Setreuid(0,0)", fn: func() error { return syscall.Setreuid(0, 0) }, filter: "Uid:", expect: "\t0\t0\t0\t0"},

		{call: "Setresgid(101,0,102)", fn: func() error { return syscall.Setresgid(101, 0, 102) }, filter: "Gid:", expect: "\t101\t0\t102\t0"},
		{call: "Setresgid(0,102,101)", fn: func() error { return syscall.Setresgid(0, 102, 101) }, filter: "Gid:", expect: "\t0\t102\t101\t102"},
		{call: "Setresgid(0,0,0)", fn: func() error { return syscall.Setresgid(0, 0, 0) }, filter: "Gid:", expect: "\t0\t0\t0\t0"},

		{call: "Setresuid(1,0,2)", fn: func() error { return syscall.Setresuid(1, 0, 2) }, filter: "Uid:", expect: "\t1\t0\t2\t0"},
		{call: "Setresuid(0,2,1)", fn: func() error { return syscall.Setresuid(0, 2, 1) }, filter: "Uid:", expect: "\t0\t2\t1\t2"},
		{call: "Setresuid(0,0,0)", fn: func() error { return syscall.Setresuid(0, 0, 0) }, filter: "Uid:", expect: "\t0\t0\t0\t0"},
	}

	for i, v := range vs {
		if err := v.fn(); err != nil {
			t.Errorf("[%d] %q failed: %v", i, v.call, err)
			continue
		}
		if err := compareStatus(v.filter, v.expect); err != nil {
			t.Errorf("[%d] %q comparison: %v", i, v.call, err)
		}
	}
}
