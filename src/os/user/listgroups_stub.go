// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build android

package user

import (
	"errors"
)

func init() {
	groupListImplemented = false
}

func listGroups(*User) ([]string, error) {
	return nil, errors.New("user: list groups not implemented")
}
