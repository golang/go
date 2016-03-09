// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin

package net

import (
	"runtime"
	"sync"
	"syscall"
	"testing"
	"time"
)

// See golang.org/issue/14548.
func TestTCPSupriousConnSetupCompletion(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	ln, err := newLocalListener("tcp")
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	wg.Add(1)
	go func(ln Listener) {
		defer wg.Done()
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			wg.Add(1)
			go func(c Conn) {
				var b [1]byte
				c.Read(b[:])
				c.Close()
				wg.Done()
			}(c)
		}
	}(ln)

	attempts := int(1e4) // larger is better
	wg.Add(attempts)
	throttle := make(chan struct{}, runtime.GOMAXPROCS(-1)*2)
	for i := 0; i < attempts; i++ {
		throttle <- struct{}{}
		go func(i int) {
			defer func() {
				<-throttle
				wg.Done()
			}()
			d := Dialer{Timeout: 50 * time.Millisecond}
			c, err := d.Dial(ln.Addr().Network(), ln.Addr().String())
			if err != nil {
				if perr := parseDialError(err); perr != nil {
					t.Errorf("#%d: %v", i, err)
				}
				return
			}
			var b [1]byte
			if _, err := c.Write(b[:]); err != nil {
				if perr := parseWriteError(err); perr != nil {
					t.Errorf("#%d: %v", i, err)
				}
				if samePlatformError(err, syscall.ENOTCONN) {
					t.Errorf("#%d: %v", i, err)
				}
			}
			c.Close()
		}(i)
	}

	ln.Close()
	wg.Wait()
}
