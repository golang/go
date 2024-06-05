// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(linux || (darwin && !ios) || dragonfly || freebsd || solaris || windows)

package net

import "io"

const supportsSendfile = false

func sendFile(c *netFD, r io.Reader) (n int64, err error, handled bool) {
	return 0, nil, false
}
