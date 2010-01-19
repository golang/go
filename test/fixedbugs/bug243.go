// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"net"
	"os"
)

func main() {
	os.Stdout.Close()
	var listen, _ = net.Listen("tcp", ":0")

	go func() {
		for {
			var conn, _ = listen.Accept()
			fmt.Println("[SERVER] ", conn)
		}
	}()

	var conn, _ = net.Dial("tcp", "", listen.Addr().String())
	fmt.Println("[CLIENT] ", conn)
}
