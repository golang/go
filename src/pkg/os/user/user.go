// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package user allows user account lookups by name or id.
package user

import (
	"strconv"
)

var implemented = false // set to true by lookup_unix.go's init

// User represents a user account.
type User struct {
	Uid      int // user id
	Gid      int // primary group id
	Username string
	Name     string
	HomeDir  string
}

// UnknownUserIdError is returned by LookupId when
// a user cannot be found.
type UnknownUserIdError int

func (e UnknownUserIdError) String() string {
	return "user: unknown userid " + strconv.Itoa(int(e))
}

// UnknownUserError is returned by Lookup when
// a user cannot be found.
type UnknownUserError string

func (e UnknownUserError) String() string {
	return "user: unknown user " + string(e)
}
