// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The toolexec program is a helper program for cmd/cover tests.
// It is used so that the go tool will call the newly built version
// of the cover program, rather than the installed one.
//
// The tests arrange to run the go tool with the argument
//    -toolexec="/path/to/toolexec /path/to/testcover"
// The go tool will invoke this program (compiled into /path/to/toolexec)
// with the arguments shown above followed by the command to run.
// This program will check whether it is expected to run the cover
// program, and if so replace it with /path/to/testcover.
package main

import (
	"os"
	"os/exec"
	"strings"
)

func main() {
	if strings.HasSuffix(strings.TrimSuffix(os.Args[2], ".exe"), "cover") {
		os.Args[2] = os.Args[1]
	}
	cmd := exec.Command(os.Args[2], os.Args[3:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		os.Exit(1)
	}
}
