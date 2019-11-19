// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"os"
)

// If the ifindex is zero, interfaceTable returns mappings of all
// network interfaces. Otherwise it returns a mapping of a specific
// interface.
func interfaceTable(ifindex int) ([]Interface, error) {
	if ifindex == 0 {
		n, err := interfaceCount()
		if err != nil {
			return nil, err
		}
		ifcs := make([]Interface, n)
		for i := range ifcs {
			ifc, err := readInterface(i)
			if err != nil {
				return nil, err
			}
			ifcs[i] = *ifc
		}
		return ifcs, nil
	}

	ifc, err := readInterface(ifindex - 1)
	if err != nil {
		return nil, err
	}
	return []Interface{*ifc}, nil
}

func readInterface(i int) (*Interface, error) {
	ifc := &Interface{
		Index: i + 1,                        // Offset the index by one to suit the contract
		Name:  netdir + "/ipifc/" + itoa(i), // Name is the full path to the interface path in plan9
	}

	ifcstat := ifc.Name + "/status"
	ifcstatf, err := open(ifcstat)
	if err != nil {
		return nil, err
	}
	defer ifcstatf.close()

	line, ok := ifcstatf.readLine()
	if !ok {
		return nil, errors.New("invalid interface status file: " + ifcstat)
	}

	fields := getFields(line)
	if len(fields) < 4 {
		return nil, errors.New("invalid interface status file: " + ifcstat)
	}

	device := fields[1]
	mtustr := fields[3]

	mtu, _, ok := dtoi(mtustr)
	if !ok {
		return nil, errors.New("invalid status file of interface: " + ifcstat)
	}
	ifc.MTU = mtu

	// Not a loopback device
	if device != "/dev/null" {
		deviceaddrf, err := open(device + "/addr")
		if err != nil {
			return nil, err
		}
		defer deviceaddrf.close()

		line, ok = deviceaddrf.readLine()
		if !ok {
			return nil, errors.New("invalid address file for interface: " + device + "/addr")
		}

		if len(line) > 0 && len(line)%2 == 0 {
			ifc.HardwareAddr = make([]byte, len(line)/2)
			var ok bool
			for i := range ifc.HardwareAddr {
				j := (i + 1) * 2
				ifc.HardwareAddr[i], ok = xtoi2(line[i*2:j], 0)
				if !ok {
					ifc.HardwareAddr = ifc.HardwareAddr[:i]
					break
				}
			}
		}

		ifc.Flags = FlagUp | FlagBroadcast | FlagMulticast
	} else {
		ifc.Flags = FlagUp | FlagMulticast | FlagLoopback
	}

	return ifc, nil
}

func interfaceCount() (int, error) {
	d, err := os.Open(netdir + "/ipifc")
	if err != nil {
		return -1, err
	}
	defer d.Close()

	names, err := d.Readdirnames(0)
	if err != nil {
		return -1, err
	}

	// Assumes that numbered files in ipifc are strictly
	// the incrementing numbered directories for the
	// interfaces
	c := 0
	for _, name := range names {
		if _, _, ok := dtoi(name); !ok {
			continue
		}
		c++
	}

	return c, nil
}

// If the ifi is nil, interfaceAddrTable returns addresses for all
// network interfaces. Otherwise it returns addresses for a specific
// interface.
func interfaceAddrTable(ifi *Interface) ([]Addr, error) {
	var ifcs []Interface
	if ifi == nil {
		var err error
		ifcs, err = interfaceTable(0)
		if err != nil {
			return nil, err
		}
	} else {
		ifcs = []Interface{*ifi}
	}

	var addrs []Addr
	for _, ifc := range ifcs {
		status := ifc.Name + "/status"
		statusf, err := open(status)
		if err != nil {
			return nil, err
		}
		defer statusf.close()

		// Read but ignore first line as it only contains the table header.
		// See https://9p.io/magic/man2html/3/ip
		if _, ok := statusf.readLine(); !ok {
			return nil, errors.New("cannot read header line for interface: " + status)
		}

		for line, ok := statusf.readLine(); ok; line, ok = statusf.readLine() {
			fields := getFields(line)
			if len(fields) < 1 {
				return nil, errors.New("cannot parse IP address for interface: " + status)
			}
			addr := fields[0]
			ip := ParseIP(addr)
			if ip == nil {
				return nil, errors.New("cannot parse IP address for interface: " + status)
			}

			// The mask is represented as CIDR relative to the IPv6 address.
			// Plan 9 internal representation is always IPv6.
			maskfld := fields[1]
			maskfld = maskfld[1:]
			pfxlen, _, ok := dtoi(maskfld)
			if !ok {
				return nil, errors.New("cannot parse network mask for interface: " + status)
			}
			var mask IPMask
			if ip.To4() != nil { // IPv4 or IPv6 IPv4-mapped address
				mask = CIDRMask(pfxlen-8*len(v4InV6Prefix), 8*IPv4len)
			}
			if ip.To16() != nil && ip.To4() == nil { // IPv6 address
				mask = CIDRMask(pfxlen, 8*IPv6len)
			}

			addrs = append(addrs, &IPNet{IP: ip, Mask: mask})
		}
	}

	return addrs, nil
}

// interfaceMulticastAddrTable returns addresses for a specific
// interface.
func interfaceMulticastAddrTable(ifi *Interface) ([]Addr, error) {
	return nil, nil
}
