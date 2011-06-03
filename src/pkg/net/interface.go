// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Network interface identification

package net

import (
	"bytes"
	"fmt"
	"os"
	"syscall"
)

// A HardwareAddr represents a physical hardware address.
type HardwareAddr []byte

func (a HardwareAddr) String() string {
	var buf bytes.Buffer
	for i, b := range a {
		if i > 0 {
			buf.WriteByte(':')
		}
		fmt.Fprintf(&buf, "%02x", b)
	}
	return buf.String()
}

// Interface represents a mapping between network interface name
// and index.  It also represents network interface facility
// information.
type Interface struct {
	Index        int          // positive integer that starts at one, zero is never used
	MTU          int          // maximum transmission unit
	Name         string       // e.g., "en0", "lo0", "eth0.100"
	HardwareAddr HardwareAddr // IEEE MAC-48, EUI-48 and EUI-64 form
	rawFlags     int
}

// IsUp returns true if ifi is up.
func (ifi *Interface) IsUp() bool {
	if ifi == nil {
		return false
	}
	return ifi.rawFlags&syscall.IFF_UP != 0
}

// IsLoopback returns true if ifi is a loopback interface.
func (ifi *Interface) IsLoopback() bool {
	if ifi == nil {
		return false
	}
	return ifi.rawFlags&syscall.IFF_LOOPBACK != 0
}

// CanBroadcast returns true if ifi supports a broadcast access
// capability.
func (ifi *Interface) CanBroadcast() bool {
	if ifi == nil {
		return false
	}
	return ifi.rawFlags&syscall.IFF_BROADCAST != 0
}

// IsPointToPoint returns true if ifi belongs to a point-to-point
// link.
func (ifi *Interface) IsPointToPoint() bool {
	if ifi == nil {
		return false
	}
	return ifi.rawFlags&syscall.IFF_POINTOPOINT != 0
}

// CanMulticast returns true if ifi supports a multicast access
// capability.
func (ifi *Interface) CanMulticast() bool {
	if ifi == nil {
		return false
	}
	return ifi.rawFlags&syscall.IFF_MULTICAST != 0
}

// Addrs returns interface addresses for a specific interface.
func (ifi *Interface) Addrs() ([]Addr, os.Error) {
	if ifi == nil {
		return nil, os.NewError("net: invalid interface")
	}
	return interfaceAddrTable(ifi.Index)
}

// Interfaces returns a list of the systems's network interfaces.
func Interfaces() ([]Interface, os.Error) {
	return interfaceTable(0)
}

// InterfaceAddrs returns a list of the system's network interface
// addresses.
func InterfaceAddrs() ([]Addr, os.Error) {
	return interfaceAddrTable(0)
}

// InterfaceByIndex returns the interface specified by index.
func InterfaceByIndex(index int) (*Interface, os.Error) {
	if index <= 0 {
		return nil, os.NewError("net: invalid interface index")
	}
	ift, err := interfaceTable(index)
	if err != nil {
		return nil, err
	}
	for _, ifi := range ift {
		return &ifi, nil
	}
	return nil, os.NewError("net: no such interface")
}

// InterfaceByName returns the interface specified by name.
func InterfaceByName(name string) (*Interface, os.Error) {
	if name == "" {
		return nil, os.NewError("net: invalid interface name")
	}
	ift, err := interfaceTable(0)
	if err != nil {
		return nil, err
	}
	for _, ifi := range ift {
		if name == ifi.Name {
			return &ifi, nil
		}
	}
	return nil, os.NewError("net: no such interface")
}
