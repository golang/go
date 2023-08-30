// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js || wasip1

package net

// GOOS=js and GOOS=wasip1 do not have typical socket networking capabilities
// found on other platforms. To help run test suites of the stdlib packages,
// an in-memory "fake network" facility is implemented.
//
// The tests in this files are intended to validate the behavior of the fake
// network stack on these platforms.

import (
	"errors"
	"syscall"
	"testing"
)

func TestFakePortExhaustion(t *testing.T) {
	if testing.Short() {
		t.Skipf("skipping test that opens 1<<16 connections")
	}

	ln := newLocalListener(t, "tcp")
	done := make(chan struct{})
	go func() {
		var accepted []Conn
		defer func() {
			for _, c := range accepted {
				c.Close()
			}
			close(done)
		}()

		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			accepted = append(accepted, c)
		}
	}()

	var dialed []Conn
	defer func() {
		ln.Close()
		for _, c := range dialed {
			c.Close()
		}
		<-done
	}()

	// Since this test is not running in parallel, we expect to be able to open
	// all 65535 valid (fake) ports. The listener is already using one, so
	// we should be able to Dial the remaining 65534.
	for len(dialed) < (1<<16)-2 {
		c, err := Dial(ln.Addr().Network(), ln.Addr().String())
		if err != nil {
			t.Fatalf("unexpected error from Dial with %v connections: %v", len(dialed), err)
		}
		dialed = append(dialed, c)
		if testing.Verbose() && len(dialed)%(1<<12) == 0 {
			t.Logf("dialed %d connections", len(dialed))
		}
	}
	t.Logf("dialed %d connections", len(dialed))

	// Now that all of the ports are in use, dialing another should fail due
	// to port exhaustion, which (for POSIX-like socket APIs) should return
	// an EADDRINUSE error.
	c, err := Dial(ln.Addr().Network(), ln.Addr().String())
	if err == nil {
		c.Close()
	}
	if errors.Is(err, syscall.EADDRINUSE) {
		t.Logf("Dial returned expected error: %v", err)
	} else {
		t.Errorf("unexpected error from Dial: %v\nwant: %v", err, syscall.EADDRINUSE)
	}

	// Opening a Listener should fail at this point too.
	ln2, err := Listen("tcp", "localhost:0")
	if err == nil {
		ln2.Close()
	}
	if errors.Is(err, syscall.EADDRINUSE) {
		t.Logf("Listen returned expected error: %v", err)
	} else {
		t.Errorf("unexpected error from Listen: %v\nwant: %v", err, syscall.EADDRINUSE)
	}

	// When we close an arbitrary connection, we should be able to reuse its port
	// even if the server hasn't yet seen the ECONNRESET for the connection.
	dialed[0].Close()
	dialed = dialed[1:]
	t.Logf("closed one connection")
	c, err = Dial(ln.Addr().Network(), ln.Addr().String())
	if err == nil {
		c.Close()
		t.Logf("Dial succeeded")
	} else {
		t.Errorf("unexpected error from Dial: %v", err)
	}
}
