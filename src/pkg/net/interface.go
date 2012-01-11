// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Network interface identification

package net

import (
	"bytes"
	"errors"
	"fmt"
)

var (
	errInvalidInterface         = errors.New("net: invalid interface")
	errInvalidInterfaceIndex    = errors.New("net: invalid interface index")
	errInvalidInterfaceName     = errors.New("net: invalid interface name")
	errNoSuchInterface          = errors.New("net: no such interface")
	errNoSuchMulticastInterface = errors.New("net: no such multicast interface")
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

// ParseMAC parses s as an IEEE 802 MAC-48, EUI-48, or EUI-64 using one of the
// following formats:
//   01:23:45:67:89:ab
//   01:23:45:67:89:ab:cd:ef
//   01-23-45-67-89-ab
//   01-23-45-67-89-ab-cd-ef
//   0123.4567.89ab
//   0123.4567.89ab.cdef
func ParseMAC(s string) (hw HardwareAddr, err error) {
	if len(s) < 14 {
		goto error
	}

	if s[2] == ':' || s[2] == '-' {
		if (len(s)+1)%3 != 0 {
			goto error
		}
		n := (len(s) + 1) / 3
		if n != 6 && n != 8 {
			goto error
		}
		hw = make(HardwareAddr, n)
		for x, i := 0, 0; i < n; i++ {
			var ok bool
			if hw[i], ok = xtoi2(s[x:], s[2]); !ok {
				goto error
			}
			x += 3
		}
	} else if s[4] == '.' {
		if (len(s)+1)%5 != 0 {
			goto error
		}
		n := 2 * (len(s) + 1) / 5
		if n != 6 && n != 8 {
			goto error
		}
		hw = make(HardwareAddr, n)
		for x, i := 0, 0; i < n; i += 2 {
			var ok bool
			if hw[i], ok = xtoi2(s[x:x+2], 0); !ok {
				goto error
			}
			if hw[i+1], ok = xtoi2(s[x+2:], s[4]); !ok {
				goto error
			}
			x += 5
		}
	} else {
		goto error
	}
	return hw, nil

error:
	return nil, errors.New("invalid MAC address: " + s)
}

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

// Interfaces returns a list of the systems's network interfaces.
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
