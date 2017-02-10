// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package exec_test

import (
	"os/user"
	"strconv"
	"syscall"
	"testing"
)

func TestCredentialNoSetGroups(t *testing.T) {
	u, err := user.Current()
	if err != nil {
		t.Fatalf("error getting current user: %v", err)
	}

	uid, err := strconv.Atoi(u.Uid)
	if err != nil {
		t.Fatalf("error converting Uid=%s to integer: %v", u.Uid, err)
	}

	gid, err := strconv.Atoi(u.Gid)
	if err != nil {
		t.Fatalf("error converting Gid=%s to integer: %v", u.Gid, err)
	}

	// If NoSetGroups is true, setgroups isn't called and cmd.Run should succeed
	cmd := helperCommand(t, "echo", "foo")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Credential: &syscall.Credential{
			Uid:         uint32(uid),
			Gid:         uint32(gid),
			NoSetGroups: true,
		},
	}

	if err = cmd.Run(); err != nil {
		t.Errorf("Failed to run command: %v", err)
	}
}
