// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux

package net

import "io"

func splice(c *netFD, r io.Reader) (int64, error, bool) {
	return 0, nil, false
}
