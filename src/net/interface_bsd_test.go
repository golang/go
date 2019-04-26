// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package net

import (
	"errors"
	"fmt"
	"os/exec"
	"runtime"
)

func (ti *testInterface) setBroadcast(vid int) error {
	if runtime.GOOS == "openbsd" {
		ti.name = fmt.Sprintf("vether%d", vid)
	} else {
		ti.name = fmt.Sprintf("vlan%d", vid)
	}
	xname, err := exec.LookPath("ifconfig")
	if err != nil {
		return err
	}
	ti.setupCmds = append(ti.setupCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ifconfig", ti.name, "create"},
	})
	ti.teardownCmds = append(ti.teardownCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ifconfig", ti.name, "destroy"},
	})
	return nil
}

func (ti *testInterface) setPointToPoint(suffix int) error {
	ti.name = fmt.Sprintf("gif%d", suffix)
	xname, err := exec.LookPath("ifconfig")
	if err != nil {
		return err
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
	return nil
}

func (ti *testInterface) setLinkLocal(suffix int) error {
	return errors.New("not yet implemented for BSD")
}
