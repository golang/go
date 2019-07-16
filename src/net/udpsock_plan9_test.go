// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/testenv"
	"runtime"
	"testing"
)

func TestListenMulticastUDP(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	ifcs, err := Interfaces()
	if err != nil {
		t.Skip(err.Error())
	}
	if len(ifcs) == 0 {
		t.Skip("no network interfaces found")
	}

	var mifc *Interface
	for _, ifc := range ifcs {
		if ifc.Flags&FlagUp|FlagMulticast != FlagUp|FlagMulticast {
			continue
		}
		mifc = &ifc
		break
	}

	if mifc == nil {
		t.Skipf("no multicast interfaces found")
	}

	c1, err := ListenMulticastUDP("udp4", mifc, &UDPAddr{IP: ParseIP("224.0.0.254")})
	if err != nil {
		t.Fatalf("multicast not working on %s", runtime.GOOS)
	}
	c1addr := c1.LocalAddr().(*UDPAddr)
	if err != nil {
		t.Fatal(err)
	}
	defer c1.Close()

	c2, err := ListenUDP("udp4", &UDPAddr{IP: IPv4zero, Port: 0})
	c2addr := c2.LocalAddr().(*UDPAddr)
	if err != nil {
		t.Fatal(err)
	}
	defer c2.Close()

	n, err := c2.WriteToUDP([]byte("data"), c1addr)
	if err != nil {
		t.Fatal(err)
	}
	if n != 4 {
		t.Fatalf("got %d; want 4", n)
	}

	n, err = c1.WriteToUDP([]byte("data"), c2addr)
	if err != nil {
		t.Fatal(err)
	}
	if n != 4 {
		t.Fatalf("got %d; want 4", n)
	}
}
