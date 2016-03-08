// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !cgo,!windows,!plan9,!android

package user

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"strconv"
)

func init() {
	userImplemented = false
	groupImplemented = false
}

func current() (*User, error) {
	u := &User{
		Uid:      currentUID(),
		Gid:      currentGID(),
		Username: os.Getenv("USER"),
		Name:     "", // ignored
		HomeDir:  os.Getenv("HOME"),
	}
	if runtime.GOOS == "nacl" {
		if u.Uid == "" {
			u.Uid = "1"
		}
		if u.Username == "" {
			u.Username = "nacl"
		}
		if u.HomeDir == "" {
			u.HomeDir = "/home/nacl"
		}
	}
	// cgo isn't available, but if we found the minimum information
	// without it, use it:
	if u.Uid != "" && u.Username != "" && u.HomeDir != "" {
		return u, nil
	}
	return u, fmt.Errorf("user: Current not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)
}

func lookupUser(username string) (*User, error) {
	return nil, errors.New("user: Lookup requires cgo")
}

func lookupUserId(uid string) (*User, error) {
	return nil, errors.New("user: LookupId requires cgo")
}

func lookupGroup(groupname string) (*Group, error) {
	return nil, errors.New("user: LookupGroup requires cgo")
}

func lookupGroupId(string) (*Group, error) {
	return nil, errors.New("user: LookupGroupId requires cgo")
}

func listGroups(*User) ([]string, error) {
	return nil, errors.New("user: GroupIds requires cgo")
}

func currentUID() string {
	if id := os.Getuid(); id >= 0 {
		return strconv.Itoa(id)
	}
	// Note: Windows returns -1, but this file isn't used on
	// Windows anyway, so this empty return path shouldn't be
	// used.
	return ""
}

func currentGID() string {
	if id := os.Getgid(); id >= 0 {
		return strconv.Itoa(id)
	}
	return ""
}
