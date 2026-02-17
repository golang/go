// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements API tests across platforms and should never have a build
// constraint.

package net

import (
	"testing"
	"time"
)

// someTimeout is used just to test that net.Conn implementations
// don't explode when their SetFooDeadline methods are called.
// It isn't actually used for testing timeouts.
const someTimeout = 1 * time.Hour

func TestConnAndListener(t *testing.T) {
	for i, network := range []string{"tcp", "unix", "unixpacket"} {
		t.Run(network, func(t *testing.T) {
			if !testableNetwork(network) {
				t.Skipf("skipping %s test", network)
			}

			ls := newLocalServer(t, network)
			defer ls.teardown()
			ch := make(chan error, 1)
			handler := func(ls *localServer, ln Listener) { ls.transponder(ln, ch) }
			if err := ls.buildup(handler); err != nil {
				t.Fatal(err)
			}
			if ls.Listener.Addr().Network() != network {
				t.Fatalf("got %s; want %s", ls.Listener.Addr().Network(), network)
			}

			c, err := Dial(ls.Listener.Addr().Network(), ls.Listener.Addr().String())
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			if c.LocalAddr().Network() != network || c.RemoteAddr().Network() != network {
				t.Fatalf("got %s->%s; want %s->%s", c.LocalAddr().Network(), c.RemoteAddr().Network(), network, network)
			}
			c.SetDeadline(time.Now().Add(someTimeout))
			c.SetReadDeadline(time.Now().Add(someTimeout))
			c.SetWriteDeadline(time.Now().Add(someTimeout))

			if _, err := c.Write([]byte("CONN AND LISTENER TEST")); err != nil {
				t.Fatal(err)
			}
			rb := make([]byte, 128)
			if _, err := c.Read(rb); err != nil {
				t.Fatal(err)
			}

			for err := range ch {
				t.Errorf("#%d: %v", i, err)
			}
		})
	}
}
