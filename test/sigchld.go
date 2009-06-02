// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "syscall"

func main() {
	syscall.Syscall(syscall.SYS_KILL, uintptr(syscall.Getpid()), syscall.SIGCHLD, 0);
	println("survived SIGCHLD");
}
