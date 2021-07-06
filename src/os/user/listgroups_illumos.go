// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo && !osusergo
// +build cgo,!osusergo

// Even though this file requires no C, it is used to provide a
// listGroup stub because all the other illumos calls work.  Otherwise,
// this stub will conflict with the lookup_stubs.go fallback.

package user

import "fmt"

func listGroups(u *User) ([]string, error) {
	return nil, fmt.Errorf("user: list groups for %s: not supported on illumos", u.Username)
}
