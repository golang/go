// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"runtime"
	"testing"
)

func TestWriteToUDP(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Logf("skipping test on %q", runtime.GOOS)
		return
	}

	l, err := ListenPacket("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}
	defer l.Close()

	testWriteToConn(t, l.LocalAddr().String())
	testWriteToPacketConn(t, l.LocalAddr().String())
}

func testWriteToConn(t *testing.T, raddr string) {
	c, err := Dial("udp", raddr)
	if err != nil {
		t.Fatalf("Dial failed: %v", err)
	}
	defer c.Close()

	ra, err := ResolveUDPAddr("udp", raddr)
	if err != nil {
		t.Fatalf("ResolveUDPAddr failed: %v", err)
	}

	_, err = c.(*UDPConn).WriteToUDP([]byte("Connection-oriented mode socket"), ra)
	if err == nil {
		t.Fatal("WriteToUDP should fail")
	}
	if err != nil && err.(*OpError).Err != ErrWriteToConnected {
		t.Fatalf("WriteToUDP should fail as ErrWriteToConnected: %v", err)
	}

	_, err = c.(*UDPConn).WriteTo([]byte("Connection-oriented mode socket"), ra)
	if err == nil {
		t.Fatal("WriteTo should fail")
	}
	if err != nil && err.(*OpError).Err != ErrWriteToConnected {
		t.Fatalf("WriteTo should fail as ErrWriteToConnected: %v", err)
	}

	_, err = c.Write([]byte("Connection-oriented mode socket"))
	if err != nil {
		t.Fatalf("Write failed: %v", err)
	}
}

func testWriteToPacketConn(t *testing.T, raddr string) {
	c, err := ListenPacket("udp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("ListenPacket failed: %v", err)
	}
	defer c.Close()

	ra, err := ResolveUDPAddr("udp", raddr)
	if err != nil {
		t.Fatalf("ResolveUDPAddr failed: %v", err)
	}

	_, err = c.(*UDPConn).WriteToUDP([]byte("Connection-less mode socket"), ra)
	if err != nil {
		t.Fatalf("WriteToUDP failed: %v", err)
	}

	_, err = c.WriteTo([]byte("Connection-less mode socket"), ra)
	if err != nil {
		t.Fatalf("WriteTo failed: %v", err)
	}

	_, err = c.(*UDPConn).Write([]byte("Connection-less mode socket"))
	if err == nil {
		t.Fatal("Write should fail")
	}
}
