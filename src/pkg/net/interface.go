// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Network interface identification

package net

import "errors"

var (
	errInvalidInterface         = errors.New("net: invalid interface")
	errInvalidInterfaceIndex    = errors.New("net: invalid interface index")
	errInvalidInterfaceName     = errors.New("net: invalid interface name")
	errNoSuchInterface          = errors.New("net: no such interface")
	errNoSuchMulticastInterface = errors.New("net: no such multicast interface")
)

// Interface represents a mapping between network interface name
// and index.  It also represents network interface facility
// information.
type Interface struct {
	Index        int          // positive integer that starts at one, zero is never used
	MTU          int          // maximum transmission unit
	Name         string       // e.g., "en0", "lo0", "eth0.100"
	HardwareAddr HardwareAddr // IEEE MAC-48, EUI-48 and EUI-64 form
	Flags        Flags        // e.g., FlagUp, FlagLoopback, FlagMulticast
}

type Flags uint

const (
	FlagUp           Flags = 1 << iota // interface is up
	FlagBroadcast                      // interface supports broadcast access capability
	FlagLoopback                       // interface is a loopback interface
	FlagPointToPoint                   // interface belongs to a point-to-point link
	FlagMulticast                      // interface supports multicast access capability
)

var flagNames = []string{
	"up",
	"broadcast",
	"loopback",
	"pointtopoint",
	"multicast",
}

func (f Flags) String() string {
	s := ""
	for i, name := range flagNames {
		if f&(1<<uint(i)) != 0 {
			if s != "" {
				s += "|"
			}
			s += name
		}
	}
	if s == "" {
		s = "0"
	}
	return s
}

// Addrs returns interface addresses for a specific interface.
func (ifi *Interface) Addrs() ([]Addr, error) {
	if ifi == nil {
		return nil, errInvalidInterface
	}
	return interfaceAddrTable(ifi.Index)
}

// MulticastAddrs returns multicast, joined group addresses for
// a specific interface.
func (ifi *Interface) MulticastAddrs() ([]Addr, error) {
	if ifi == nil {
		return nil, errInvalidInterface
	}
	return interfaceMulticastAddrTable(ifi.Index)
}

// Interfaces returns a list of the system's network interfaces.
func Interfaces() ([]Interface, error) {
	return interfaceTable(0)
}

// InterfaceAddrs returns a list of the system's network interface
// addresses.
func InterfaceAddrs() ([]Addr, error) {
	return interfaceAddrTable(0)
}

// InterfaceByIndex returns the interface specified by index.
func InterfaceByIndex(index int) (*Interface, error) {
	if index <= 0 {
		return nil, errInvalidInterfaceIndex
	}
	ift, err := interfaceTable(index)
	if err != nil {
		return nil, err
	}
	for _, ifi := range ift {
		return &ifi, nil
	}
	return nil, errNoSuchInterface
}

// InterfaceByName returns the interface specified by name.
func InterfaceByName(name string) (*Interface, error) {
	if name == "" {
		return nil, errInvalidInterfaceName
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
	return nil, errNoSuchInterface
}
