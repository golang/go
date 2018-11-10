// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"syscall"
	"testing"
)

func TestRawConn(t *testing.T) {
	c, err := newLocalPacketListener("udp")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	cc, err := c.(*UDPConn).SyscallConn()
	if err != nil {
		t.Fatal(err)
	}

	var operr error
	fn := func(s uintptr) {
		operr = syscall.SetsockoptInt(syscall.Handle(s), syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1)
	}
	err = cc.Control(fn)
	if err != nil || operr != nil {
		t.Fatal(err, operr)
	}
	c.Close()
	err = cc.Control(fn)
	if err == nil {
		t.Fatal("should fail")
	}
}
