// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"net"
)

func init() {
	registerInit("NetpollDeadlock", NetpollDeadlockInit)
	register("NetpollDeadlock", NetpollDeadlock)
}

func NetpollDeadlockInit() {
	fmt.Println("dialing")
	c, err := net.Dial("tcp", "localhost:14356")
	if err == nil {
		c.Close()
	} else {
		fmt.Println("error: ", err)
	}
}

func NetpollDeadlock() {
	fmt.Println("done")
}
