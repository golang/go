// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"sync"
	"time"
)

var (
	errInvalidInterface         = errors.New("invalid network interface")
	errInvalidInterfaceIndex    = errors.New("invalid network interface index")
	errInvalidInterfaceName     = errors.New("invalid network interface name")
	errNoSuchInterface          = errors.New("no such network interface")
	errNoSuchMulticastInterface = errors.New("no such multicast network interface")
)

// Interface represents a mapping between network interface name
// and index. It also represents network interface facility
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
		return nil, &OpError{Op: "route", Net: "ip+net", Source: nil, Addr: nil, Err: errInvalidInterface}
	}
	ifat, err := interfaceAddrTable(ifi)
	if err != nil {
		err = &OpError{Op: "route", Net: "ip+net", Source: nil, Addr: nil, Err: err}
	}
	return ifat, err
}

// MulticastAddrs returns multicast, joined group addresses for
// a specific interface.
func (ifi *Interface) MulticastAddrs() ([]Addr, error) {
	if ifi == nil {
		return nil, &OpError{Op: "route", Net: "ip+net", Source: nil, Addr: nil, Err: errInvalidInterface}
	}
	ifat, err := interfaceMulticastAddrTable(ifi)
	if err != nil {
		err = &OpError{Op: "route", Net: "ip+net", Source: nil, Addr: nil, Err: err}
	}
	return ifat, err
}

// Interfaces returns a list of the system's network interfaces.
func Interfaces() ([]Interface, error) {
	ift, err := interfaceTable(0)
	if err != nil {
		return nil, &OpError{Op: "route", Net: "ip+net", Source: nil, Addr: nil, Err: err}
	}
	if len(ift) != 0 {
		zoneCache.update(ift)
	}
	return ift, nil
}

// InterfaceAddrs returns a list of the system's network interface
// addresses.
func InterfaceAddrs() ([]Addr, error) {
	ifat, err := interfaceAddrTable(nil)
	if err != nil {
		err = &OpError{Op: "route", Net: "ip+net", Source: nil, Addr: nil, Err: err}
	}
	return ifat, err
}

// InterfaceByIndex returns the interface specified by index.
func InterfaceByIndex(index int) (*Interface, error) {
	if index <= 0 {
		return nil, &OpError{Op: "route", Net: "ip+net", Source: nil, Addr: nil, Err: errInvalidInterfaceIndex}
	}
	ift, err := interfaceTable(index)
	if err != nil {
		return nil, &OpError{Op: "route", Net: "ip+net", Source: nil, Addr: nil, Err: err}
	}
	ifi, err := interfaceByIndex(ift, index)
	if err != nil {
		err = &OpError{Op: "route", Net: "ip+net", Source: nil, Addr: nil, Err: err}
	}
	return ifi, err
}

func interfaceByIndex(ift []Interface, index int) (*Interface, error) {
	for _, ifi := range ift {
		if index == ifi.Index {
			return &ifi, nil
		}
	}
	return nil, errNoSuchInterface
}

// InterfaceByName returns the interface specified by name.
func InterfaceByName(name string) (*Interface, error) {
	if name == "" {
		return nil, &OpError{Op: "route", Net: "ip+net", Source: nil, Addr: nil, Err: errInvalidInterfaceName}
	}
	ift, err := interfaceTable(0)
	if err != nil {
		return nil, &OpError{Op: "route", Net: "ip+net", Source: nil, Addr: nil, Err: err}
	}
	if len(ift) != 0 {
		zoneCache.update(ift)
	}
	for _, ifi := range ift {
		if name == ifi.Name {
			return &ifi, nil
		}
	}
	return nil, &OpError{Op: "route", Net: "ip+net", Source: nil, Addr: nil, Err: errNoSuchInterface}
}

// An ipv6ZoneCache represents a cache holding partial network
// interface information. It is used for reducing the cost of IPv6
// addressing scope zone resolution.
type ipv6ZoneCache struct {
	sync.RWMutex                // guard the following
	lastFetched  time.Time      // last time routing information was fetched
	toIndex      map[string]int // interface name to its index
	toName       map[int]string // interface index to its name
}

var zoneCache = ipv6ZoneCache{
	toIndex: make(map[string]int),
	toName:  make(map[int]string),
}

func (zc *ipv6ZoneCache) update(ift []Interface) {
	zc.Lock()
	defer zc.Unlock()
	now := time.Now()
	if zc.lastFetched.After(now.Add(-60 * time.Second)) {
		return
	}
	zc.lastFetched = now
	if len(ift) == 0 {
		var err error
		if ift, err = interfaceTable(0); err != nil {
			return
		}
	}
	zc.toIndex = make(map[string]int, len(ift))
	zc.toName = make(map[int]string, len(ift))
	for _, ifi := range ift {
		zc.toIndex[ifi.Name] = ifi.Index
		zc.toName[ifi.Index] = ifi.Name
	}
}

func zoneToString(zone int) string {
	if zone == 0 {
		return ""
	}
	zoneCache.update(nil)
	zoneCache.RLock()
	defer zoneCache.RUnlock()
	name, ok := zoneCache.toName[zone]
	if !ok {
		name = uitoa(uint(zone))
	}
	return name
}

func zoneToInt(zone string) int {
	if zone == "" {
		return 0
	}
	zoneCache.update(nil)
	zoneCache.RLock()
	defer zoneCache.RUnlock()
	index, ok := zoneCache.toIndex[zone]
	if !ok {
		index, _, _ = dtoi(zone, 0)
	}
	return index
}
