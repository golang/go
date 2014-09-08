// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package user

import (
	"fmt"
	"io/ioutil"
	"os"
	"syscall"
)

// Partial os/user support on Plan 9.
// Supports Current(), but not Lookup()/LookupId().
// The latter two would require parsing /adm/users.
const (
	userFile = "/dev/user"
)

func current() (*User, error) {
	ubytes, err := ioutil.ReadFile(userFile)
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

func lookup(username string) (*User, error) {
	return nil, syscall.EPLAN9
}

func lookupId(uid string) (*User, error) {
	return nil, syscall.EPLAN9
}
