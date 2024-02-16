// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io"
	"log"
	"net"
	"runtime/internal/atomic"
	"sync"
	"time"
	_ "unsafe" // for go:linkname
)

// The bug is that netpollWaiters increases monotonically.
// This doesn't cause a problem until it overflows.
// Use linkname to see the value.
//
//go:linkname netpollWaiters runtime.netpollWaiters
var netpollWaiters atomic.Uint32

func init() {
	register("NetpollWaiters", NetpollWaiters)
}

func NetpollWaiters() {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		log.Fatal(err)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		conn, err := listener.Accept()
		if err != nil {
			log.Fatal(err)
		}
		defer conn.Close()
		if _, err := io.Copy(io.Discard, conn); err != nil {
			log.Fatal(err)
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		conn, err := net.Dial("tcp", listener.Addr().String())
		if err != nil {
			log.Fatal(err)
		}
		defer conn.Close()
		for i := 0; i < 10; i++ {
			fmt.Fprintf(conn, "%d\n", i)
			time.Sleep(time.Millisecond)
		}
	}()

	wg.Wait()
	if v := netpollWaiters.Load(); v != 0 {
		log.Fatalf("current waiters %v", v)
	}

	fmt.Println("OK")
}
