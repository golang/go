// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package net

import (
	"bytes"
	"syscall"
	"testing"
	"time"
)

// Test that a client can't trigger an endless loop of write system
// calls on the server by shutting down the write side on the client.
// Possibility raised in the discussion of https://golang.org/cl/71973.
func TestEndlessWrite(t *testing.T) {
	t.Parallel()
	c := make(chan bool)
	server := func(cs *TCPConn) error {
		cs.CloseWrite()
		<-c
		return nil
	}
	client := func(ss *TCPConn) error {
		// Tell the server to return when we return.
		defer close(c)

		// Loop writing to the server. The server is not reading
		// anything, so this will eventually block, and then time out.
		b := bytes.Repeat([]byte{'a'}, 8192)
		cagain := 0
		for {
			n, err := ss.conn.fd.pfd.WriteOnce(b)
			if n > 0 {
				cagain = 0
			}
			switch err {
			case nil:
			case syscall.EAGAIN:
				if cagain == 0 {
					// We've written enough data to
					// start blocking. Set a deadline
					// so that we will stop.
					ss.SetWriteDeadline(time.Now().Add(5 * time.Millisecond))
				}
				cagain++
				if cagain > 20 {
					t.Error("looping on EAGAIN")
					return nil
				}
				if err = ss.conn.fd.pfd.WaitWrite(); err != nil {
					t.Logf("client WaitWrite: %v", err)
					return nil
				}
			default:
				// We expect to eventually get an error.
				t.Logf("client WriteOnce: %v", err)
				return nil
			}
		}
	}
	withTCPConnPair(t, client, server)
}
