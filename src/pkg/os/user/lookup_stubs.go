// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build openbsd plan9 windows

package user

import (
	"fmt"
	"os"
	"runtime"
)

func Lookup(username string) (*User, os.Error) {
	return nil, fmt.Errorf("user: Lookup not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)
}

func LookupId(int) (*User, os.Error) {
	return nil, fmt.Errorf("user: LookupId not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)
}
