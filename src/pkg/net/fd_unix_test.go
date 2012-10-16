// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

package net

import (
	"testing"
)

// Issue 3590. netFd.AddFD should return an error 
// from the underlying pollster rather than panicing.
func TestAddFDReturnsError(t *testing.T) {
	l, err := Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()

	go func() {
		for {
			c, err := l.Accept()
			if err != nil {
				return
			}
			defer c.Close()
		}
	}()

	c, err := Dial("tcp", l.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	// replace c's pollServer with a closed version.
	ps, err := newPollServer()
	if err != nil {
		t.Fatal(err)
	}
	ps.poll.Close()
	c.(*TCPConn).conn.fd.pollServer = ps

	var b [1]byte
	_, err = c.Read(b[:])
	if err, ok := err.(*OpError); ok {
		if err.Op == "addfd" {
			return
		}
		if err, ok := err.Err.(*OpError); ok {
			// the err is sometimes wrapped by another OpError
			if err.Op == "addfd" {
				return
			}
		}
	}
	t.Error(err)
}
