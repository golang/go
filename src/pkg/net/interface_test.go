// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"testing"
)

func sameInterface(i, j *Interface) bool {
	if i == nil || j == nil {
		return false
	}
	if i.Index == j.Index && i.Name == j.Name && bytes.Equal(i.HardwareAddr, j.HardwareAddr) {
		return true
	}
	return false
}

func interfaceFlagsString(ifi *Interface) string {
	fs := "<"
	if ifi.IsUp() {
		fs += "UP,"
	}
	if ifi.CanBroadcast() {
		fs += "BROADCAST,"
	}
	if ifi.IsLoopback() {
		fs += "LOOPBACK,"
	}
	if ifi.IsPointToPoint() {
		fs += "POINTOPOINT,"
	}
	if ifi.CanMulticast() {
		fs += "MULTICAST,"
	}
	if len(fs) > 1 {
		fs = fs[:len(fs)-1]
	}
	fs += ">"
	return fs
}

func TestInterfaces(t *testing.T) {
	ift, err := Interfaces()
	if err != nil {
		t.Fatalf("Interfaces() failed: %v", err)
	}
	t.Logf("table: len/cap = %v/%v\n", len(ift), cap(ift))

	for _, ifi := range ift {
		ifxi, err := InterfaceByIndex(ifi.Index)
		if err != nil {
			t.Fatalf("InterfaceByIndex(%#q) failed: %v", ifi.Index, err)
		}
		if !sameInterface(ifxi, &ifi) {
			t.Fatalf("InterfaceByIndex(%#q) = %v, want %v", ifi.Index, *ifxi, ifi)
		}
		ifxn, err := InterfaceByName(ifi.Name)
		if err != nil {
			t.Fatalf("InterfaceByName(%#q) failed: %v", ifi.Name, err)
		}
		if !sameInterface(ifxn, &ifi) {
			t.Fatalf("InterfaceByName(%#q) = %v, want %v", ifi.Name, *ifxn, ifi)
		}
		ifat, err := ifi.Addrs()
		if err != nil {
			t.Fatalf("Interface.Addrs() failed: %v", err)
		}
		t.Logf("%s: flags %s, ifindex %v, mtu %v\n", ifi.Name, interfaceFlagsString(&ifi), ifi.Index, ifi.MTU)
		for _, ifa := range ifat {
			t.Logf("\tinterface address %s\n", ifa.String())
		}
		t.Logf("\thardware address %v", ifi.HardwareAddr.String())
	}
}

func TestInterfaceAddrs(t *testing.T) {
	ifat, err := InterfaceAddrs()
	if err != nil {
		t.Fatalf("InterfaceAddrs() failed: %v", err)
	}
	t.Logf("table: len/cap = %v/%v\n", len(ifat), cap(ifat))

	for _, ifa := range ifat {
		t.Logf("interface address %s\n", ifa.String())
	}
}
