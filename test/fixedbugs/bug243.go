// [ $GOOS != nacl ] || exit 0  # no network
// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"net"
)

func main() {
	var listen, _ = net.Listen("tcp", "127.0.0.1:0")

	go func() {
		for {
			var conn, _ = listen.Accept()
			_ = conn
		}
	}()

	var conn, _ = net.Dial("tcp", "", listen.Addr().String())
	_ = conn
}
