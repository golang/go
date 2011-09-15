// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd openbsd

package net

import (
	"io"
	"os"
)

func sendFile(c *netFD, r io.Reader) (n int64, err os.Error, handled bool) {
	return 0, nil, false
}
