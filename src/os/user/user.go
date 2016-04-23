// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package user allows user account lookups by name or id.
package user

import (
	"strconv"
)

var (
	userImplemented  = true // set to false by lookup_stubs.go's init
	groupImplemented = true // set to false by lookup_stubs.go's init
)

// User represents a user account.
//
// On POSIX systems Uid and Gid contain a decimal number
// representing uid and gid. On windows Uid and Gid
// contain security identifier (SID) in a string format.
// On Plan 9, Uid, Gid, Username, and Name will be the
// contents of /dev/user.
type User struct {
	Uid      string // user ID
	Gid      string // primary group ID
	Username string
	Name     string
	HomeDir  string
}

// Group represents a grouping of users.
//
// On POSIX systems Gid contains a decimal number
// representing the group ID.
type Group struct {
	Gid  string // group ID
	Name string // group name
}

// UnknownUserIdError is returned by LookupId when
// a user cannot be found.
type UnknownUserIdError int

func (e UnknownUserIdError) Error() string {
	return "user: unknown userid " + strconv.Itoa(int(e))
}

// UnknownUserError is returned by Lookup when
// a user cannot be found.
type UnknownUserError string

func (e UnknownUserError) Error() string {
	return "user: unknown user " + string(e)
}

// UnknownGroupIdError is returned by LookupGroupId when
// a group cannot be found.
type UnknownGroupIdError string

func (e UnknownGroupIdError) Error() string {
	return "group: unknown groupid " + string(e)
}

// UnknownGroupError is returned by LookupGroup when
// a group cannot be found.
type UnknownGroupError string

func (e UnknownGroupError) Error() string {
	return "group: unknown group " + string(e)
}
