// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux

package pprof

import (
	"errors"
)

func linuxKernelVersion() (major, minor, patch int, err error) {
	return 0, 0, 0, errors.New("not running on linux")
}
