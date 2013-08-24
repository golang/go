// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package user allows user account lookups by name or id.
package user

import (
	"strconv"
)

var implemented = true // set to false by lookup_stubs.go's init

// User represents a user account.
//
// On posix systems Uid and Gid contain a decimal number
// representing uid and gid. On windows Uid and Gid
// contain security identifier (SID) in a string format.
// On Plan 9, Uid, Gid, Username, and Name will be the
// contents of /dev/user.
type User struct {
	Uid      string // user id
	Gid      string // primary group id
	Username string
	Name     string
	HomeDir  string
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
