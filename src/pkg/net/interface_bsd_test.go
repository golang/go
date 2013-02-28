// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd netbsd openbsd

package net

import (
	"fmt"
	"os/exec"
)

func (ti *testInterface) setBroadcast(suffix int) {
	ti.name = fmt.Sprintf("vlan%d", suffix)
	xname, err := exec.LookPath("ifconfig")
	if err != nil {
		xname = "ifconfig"
	}
	ti.setupCmds = append(ti.setupCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ifconfig", ti.name, "create"},
	})
	ti.teardownCmds = append(ti.teardownCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ifconfig", ti.name, "destroy"},
	})
}

func (ti *testInterface) setPointToPoint(suffix int, local, remote string) {
	ti.name = fmt.Sprintf("gif%d", suffix)
	ti.local = local
	ti.remote = remote
	xname, err := exec.LookPath("ifconfig")
	if err != nil {
		xname = "ifconfig"
	}
	ti.setupCmds = append(ti.setupCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ifconfig", ti.name, "create"},
	})
	ti.setupCmds = append(ti.setupCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ifconfig", ti.name, "inet", ti.local, ti.remote},
	})
	ti.teardownCmds = append(ti.teardownCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ifconfig", ti.name, "destroy"},
	})
}
