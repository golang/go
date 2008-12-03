// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "syscall"

func getpid() int64 {
	r1, r2, err := syscall.Syscall(syscall.SYS_GETPID, 0, 0, 0);
	return r1;
}

func main() {
	syscall.Syscall(syscall.SYS_KILL, getpid(), syscall.SIGCHLD, 0);
	println("survived SIGCHLD");
}
