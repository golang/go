// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"net"
	"testing"
	"time"
)

func pr3DialPair(t *testing.T) (cli, srv net.Conn) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Skipf("loopback unavailable: %v", err)
	}
	defer ln.Close()
	done := make(chan net.Conn, 1)
	go func() {
		c, err := ln.Accept()
		if err == nil {
			done <- c
		}
	}()
	cli, err = net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Skipf("loopback dial unavailable: %v", err)
	}
	select {
	case srv = <-done:
	case <-time.After(2 * time.Second):
		cli.Close()
		t.Fatalf("accept timeout")
	}
	return cli, srv
}
