// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import "testing"

func TestTCP4ListenZero(t *testing.T) {
	l, err := Listen("tcp4", "0.0.0.0:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	if a := l.Addr(); isNotIPv4(a) {
		t.Errorf("address does not contain IPv4: %v", a)
	}
}

func TestUDP4ListenZero(t *testing.T) {
	c, err := ListenPacket("udp4", "0.0.0.0:0")
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if a := c.LocalAddr(); isNotIPv4(a) {
		t.Errorf("address does not contain IPv4: %v", a)
	}
}
