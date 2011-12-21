// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd netbsd openbsd

package net

import "io"

func sendFile(c *netFD, r io.Reader) (n int64, err error, handled bool) {
	return 0, nil, false
}
