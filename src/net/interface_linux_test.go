// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"os/exec"
	"testing"
)

func (ti *testInterface) setBroadcast(suffix int) error {
	ti.name = fmt.Sprintf("gotest%d", suffix)
	xname, err := exec.LookPath("ip")
	if err != nil {
		return err
	}
	ti.setupCmds = append(ti.setupCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ip", "link", "add", ti.name, "type", "dummy"},
	})
	ti.setupCmds = append(ti.setupCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ip", "address", "add", ti.local, "peer", ti.remote, "dev", ti.name},
	})
	ti.teardownCmds = append(ti.teardownCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ip", "address", "del", ti.local, "peer", ti.remote, "dev", ti.name},
	})
	ti.teardownCmds = append(ti.teardownCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ip", "link", "delete", ti.name, "type", "dummy"},
	})
	return nil
}

func (ti *testInterface) setLinkLocal(suffix int) error {
	ti.name = fmt.Sprintf("gotest%d", suffix)
	xname, err := exec.LookPath("ip")
	if err != nil {
		return err
	}
	ti.setupCmds = append(ti.setupCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ip", "link", "add", ti.name, "type", "dummy"},
	})
	ti.setupCmds = append(ti.setupCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ip", "address", "add", ti.local, "dev", ti.name},
	})
	ti.teardownCmds = append(ti.teardownCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ip", "address", "del", ti.local, "dev", ti.name},
	})
	ti.teardownCmds = append(ti.teardownCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ip", "link", "delete", ti.name, "type", "dummy"},
	})
	return nil
}

func (ti *testInterface) setPointToPoint(suffix int) error {
	ti.name = fmt.Sprintf("gotest%d", suffix)
	xname, err := exec.LookPath("ip")
	if err != nil {
		return err
	}
	ti.setupCmds = append(ti.setupCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ip", "tunnel", "add", ti.name, "mode", "gre", "local", ti.local, "remote", ti.remote},
	})
	ti.setupCmds = append(ti.setupCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ip", "address", "add", ti.local, "peer", ti.remote, "dev", ti.name},
	})
	ti.teardownCmds = append(ti.teardownCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ip", "address", "del", ti.local, "peer", ti.remote, "dev", ti.name},
	})
	ti.teardownCmds = append(ti.teardownCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ip", "tunnel", "del", ti.name, "mode", "gre", "local", ti.local, "remote", ti.remote},
	})
	return nil
}

const (
	numOfTestIPv4MCAddrs = 14
	numOfTestIPv6MCAddrs = 18
)

var (
	igmpInterfaceTable = []Interface{
		{Name: "lo"},
		{Name: "eth0"}, {Name: "eth1"}, {Name: "eth2"},
		{Name: "eth0.100"}, {Name: "eth0.101"}, {Name: "eth0.102"}, {Name: "eth0.103"},
		{Name: "device1tap2"},
	}
	igmp6InterfaceTable = []Interface{
		{Name: "lo"},
		{Name: "eth0"}, {Name: "eth1"}, {Name: "eth2"},
		{Name: "eth0.100"}, {Name: "eth0.101"}, {Name: "eth0.102"}, {Name: "eth0.103"},
		{Name: "device1tap2"},
		{Name: "pan0"},
	}
)

func TestParseProcNet(t *testing.T) {
	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("panicked: %v", p)
		}
	}()

	var ifmat4 []Addr
	for _, ifi := range igmpInterfaceTable {
		ifmat := parseProcNetIGMP("testdata/igmp", &ifi)
		ifmat4 = append(ifmat4, ifmat...)
	}
	if len(ifmat4) != numOfTestIPv4MCAddrs {
		t.Fatalf("got %d; want %d", len(ifmat4), numOfTestIPv4MCAddrs)
	}

	var ifmat6 []Addr
	for _, ifi := range igmp6InterfaceTable {
		ifmat := parseProcNetIGMP6("testdata/igmp6", &ifi)
		ifmat6 = append(ifmat6, ifmat...)
	}
	if len(ifmat6) != numOfTestIPv6MCAddrs {
		t.Fatalf("got %d; want %d", len(ifmat6), numOfTestIPv6MCAddrs)
	}
}
