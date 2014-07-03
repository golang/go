// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !cgo,!windows,!plan9 android

package user

import (
	"fmt"
	"runtime"
)

func init() {
	implemented = false
}

func current() (*User, error) {
	return nil, fmt.Errorf("user: Current not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)
}

func lookup(username string) (*User, error) {
	return nil, fmt.Errorf("user: Lookup not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)
}

func lookupId(uid string) (*User, error) {
	return nil, fmt.Errorf("user: LookupId not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)
}
