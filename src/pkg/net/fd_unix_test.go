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
	ln := newLocalListener(t).(*TCPListener)
	defer ln.Close()
	connected := make(chan bool)
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			connected <- true
			defer c.Close()
		}
	}()

	c, err := DialTCP("tcp", nil, ln.Addr().(*TCPAddr))
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	<-connected

	// replace c's pollServer with a closed version.
	ps, err := newPollServer()
	if err != nil {
		t.Fatal(err)
	}
	ps.poll.Close()
	c.conn.fd.pollServer = ps

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
	t.Error("unexpected error:", err)
}
