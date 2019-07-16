// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo,!osusergo

package user

import "fmt"

func listGroups(u *User) ([]string, error) {
	return nil, fmt.Errorf("user: list groups for %s: not supported on AIX", u.Username)
}
