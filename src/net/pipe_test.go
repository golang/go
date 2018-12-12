// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net_test

import (
	"io"
	"net"
	"testing"
	"time"

	"internal/x/net/nettest"
)

func TestPipe(t *testing.T) {
	nettest.TestConn(t, func() (c1, c2 net.Conn, stop func(), err error) {
		c1, c2 = net.Pipe()
		stop = func() {
			c1.Close()
			c2.Close()
		}
		return
	})
}

func TestPipeCloseError(t *testing.T) {
	c1, c2 := net.Pipe()
	c1.Close()

	if _, err := c1.Read(nil); err != io.ErrClosedPipe {
		t.Errorf("c1.Read() = %v, want io.ErrClosedPipe", err)
	}
	if _, err := c1.Write(nil); err != io.ErrClosedPipe {
		t.Errorf("c1.Write() = %v, want io.ErrClosedPipe", err)
	}
	if err := c1.SetDeadline(time.Time{}); err != io.ErrClosedPipe {
		t.Errorf("c1.SetDeadline() = %v, want io.ErrClosedPipe", err)
	}
	if _, err := c2.Read(nil); err != io.EOF {
		t.Errorf("c2.Read() = %v, want io.EOF", err)
	}
	if _, err := c2.Write(nil); err != io.ErrClosedPipe {
		t.Errorf("c2.Write() = %v, want io.ErrClosedPipe", err)
	}
	if err := c2.SetDeadline(time.Time{}); err != io.ErrClosedPipe {
		t.Errorf("c2.SetDeadline() = %v, want io.ErrClosedPipe", err)
	}
}
