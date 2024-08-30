// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package user

import (
	"fmt"
	"os"
	"syscall"
)

// Partial os/user support on Plan 9.
// Supports Current(), but not Lookup()/LookupId().
// The latter two would require parsing /adm/users.

func init() {
	userImplemented = false
	groupImplemented = false
	groupListImplemented = false
}

var (
	// unused variables (in this implementation)
	// modified during test to exercise code paths in the cgo implementation.
	userBuffer  = 0
	groupBuffer = 0
)

func current() (*User, error) {
	ubytes, err := os.ReadFile("/dev/user")
	if err != nil {
		return nil, fmt.Errorf("user: %s", err)
	}

	uname := string(ubytes)

	u := &User{
		Uid:      uname,
		Gid:      uname,
		Username: uname,
		Name:     uname,
		HomeDir:  os.Getenv("home"),
	}

	return u, nil
}

func lookupUser(username string) (*User, error) {
	return nil, syscall.EPLAN9
}

func lookupUserId(uid string) (*User, error) {
	return nil, syscall.EPLAN9
}

func lookupGroup(groupname string) (*Group, error) {
	return nil, syscall.EPLAN9
}

func lookupGroupId(string) (*Group, error) {
	return nil, syscall.EPLAN9
}

func listGroups(*User) ([]string, error) {
	return nil, syscall.EPLAN9
}
