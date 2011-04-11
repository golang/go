// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"flag"
	"runtime"
	"testing"
)

var multicast = flag.Bool("multicast", false, "enable multicast tests")

func TestMulticastJoinAndLeave(t *testing.T) {
	if runtime.GOOS == "windows" {
		return
	}
	if !*multicast {
		t.Logf("test disabled; use --multicast to enable")
		return
	}

	addr := &UDPAddr{
		IP:   IPv4zero,
		Port: 0,
	}
	// open a UDPConn
	conn, err := ListenUDP("udp4", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	// try to join group
	mcast := IPv4(224, 0, 0, 254)
	err = conn.JoinGroup(mcast)
	if err != nil {
		t.Fatal(err)
	}

	// try to leave group
	err = conn.LeaveGroup(mcast)
	if err != nil {
		t.Fatal(err)
	}
}

func TestJoinFailureWithIPv6Address(t *testing.T) {
	if !*multicast {
		t.Logf("test disabled; use --multicast to enable")
		return
	}
	addr := &UDPAddr{
		IP:   IPv4zero,
		Port: 0,
	}

	// open a UDPConn
	conn, err := ListenUDP("udp4", addr)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	// try to join group
	mcast := ParseIP("ff02::1")
	err = conn.JoinGroup(mcast)
	if err == nil {
		t.Fatal("JoinGroup succeeded, should fail")
	}
	t.Logf("%s", err)
}
