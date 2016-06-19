// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build android

package user

import "errors"

func init() {
	userImplemented = false
	groupImplemented = false
}

func current() (*User, error) {
	return nil, errors.New("user: Current not implemented on android")
}

func lookupUser(string) (*User, error) {
	return nil, errors.New("user: Lookup not implemented on android")
}

func lookupUserId(string) (*User, error) {
	return nil, errors.New("user: LookupId not implemented on android")
}

func lookupGroup(string) (*Group, error) {
	return nil, errors.New("user: LookupGroup not implemented on android")
}

func lookupGroupId(string) (*Group, error) {
	return nil, errors.New("user: LookupGroupId not implemented on android")
}

func listGroups(*User) ([]string, error) {
	return nil, errors.New("user: GroupIds not implemented on android")
}
