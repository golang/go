// [ "$GOOS" == windows ] ||
// ($G $D/$F.go && $L $F.$A && ./$A.out 2>&1 | cmp - $D/$F.out)

// NOTE: This test is not run by 'run.go' and so not run by all.bash.
// To run this test you must use the ./run shell script.

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that a program can survive SIGCHLD.

package main

import "syscall"

func main() {
	syscall.Kill(syscall.Getpid(), syscall.SIGCHLD)
	println("survived SIGCHLD")
}
