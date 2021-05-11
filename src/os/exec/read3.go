// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore
// +build ignore

// This is a test program that verifies that it can read from
// descriptor 3 and that no other descriptors are open.
// This is not done via TestHelperProcess and GO_WANT_HELPER_PROCESS
// because we want to ensure that this program does not use cgo,
// because C libraries can open file descriptors behind our backs
// and confuse the test. See issue 25628.
package main

import (
	"fmt"
	"internal/poll"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

func main() {
	fd3 := os.NewFile(3, "fd3")
	bs, err := io.ReadAll(fd3)
	if err != nil {
		fmt.Printf("ReadAll from fd 3: %v\n", err)
		os.Exit(1)
	}

	// Now verify that there are no other open fds.
	// stdin == 0
	// stdout == 1
	// stderr == 2
	// descriptor from parent == 3
	// All descriptors 4 and up should be available,
	// except for any used by the network poller.
	var files []*os.File
	for wantfd := uintptr(4); wantfd <= 100; wantfd++ {
		if poll.IsPollDescriptor(wantfd) {
			continue
		}
		f, err := os.Open(os.Args[0])
		if err != nil {
			fmt.Printf("error opening file with expected fd %d: %v", wantfd, err)
			os.Exit(1)
		}
		if got := f.Fd(); got != wantfd {
			fmt.Printf("leaked parent file. fd = %d; want %d\n", got, wantfd)
			fdfile := fmt.Sprintf("/proc/self/fd/%d", wantfd)
			link, err := os.Readlink(fdfile)
			fmt.Printf("readlink(%q) = %q, %v\n", fdfile, link, err)
			var args []string
			switch runtime.GOOS {
			case "plan9":
				args = []string{fmt.Sprintf("/proc/%d/fd", os.Getpid())}
			case "aix", "solaris", "illumos":
				args = []string{fmt.Sprint(os.Getpid())}
			default:
				args = []string{"-p", fmt.Sprint(os.Getpid())}
			}

			// Determine which command to use to display open files.
			ofcmd := "lsof"
			switch runtime.GOOS {
			case "dragonfly", "freebsd", "netbsd", "openbsd":
				ofcmd = "fstat"
			case "plan9":
				ofcmd = "/bin/cat"
			case "aix":
				ofcmd = "procfiles"
			case "solaris", "illumos":
				ofcmd = "pfiles"
			}

			cmd := exec.Command(ofcmd, args...)
			out, err := cmd.CombinedOutput()
			if err != nil {
				fmt.Fprintf(os.Stderr, "%s failed: %v\n", strings.Join(cmd.Args, " "), err)
			}
			fmt.Printf("%s", out)
			os.Exit(1)
		}
		files = append(files, f)
	}

	for _, f := range files {
		f.Close()
	}

	// Referring to fd3 here ensures that it is not
	// garbage collected, and therefore closed, while
	// executing the wantfd loop above. It doesn't matter
	// what we do with fd3 as long as we refer to it;
	// closing it is the easy choice.
	fd3.Close()

	os.Stdout.Write(bs)
}
