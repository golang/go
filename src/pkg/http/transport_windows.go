// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"os"
	"net"
)

func init() {
	remoteSideClosedFunc = func(err os.Error) (out bool) {
		op, ok := err.(*net.OpError)
		if ok && op.Op == "WSARecv" && op.Net == "tcp" && op.Error == os.Errno(10058) {
			// TODO(bradfitz): find the symbol for 10058
			return true
		}
		return false
	}
}
