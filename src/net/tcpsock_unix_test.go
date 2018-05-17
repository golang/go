// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9,!windows

package net

import (
	"context"
	"internal/testenv"
	"math/rand"
	"runtime"
	"sync"
	"syscall"
	"testing"
	"time"
)

// See golang.org/issue/14548.
func TestTCPSpuriousConnSetupCompletion(t *testing.T) {
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
					t.Errorf("#%d: %v (original error: %v)", i, perr, err)
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

// Issue 19289.
// Test that a canceled Dial does not cause a subsequent Dial to succeed.
func TestTCPSpuriousConnSetupCompletionWithCancel(t *testing.T) {
	if testenv.Builder() == "" {
		testenv.MustHaveExternalNetwork(t)
	}
	defer dnsWaitGroup.Wait()
	t.Parallel()
	const tries = 10000
	var wg sync.WaitGroup
	wg.Add(tries * 2)
	sem := make(chan bool, 5)
	for i := 0; i < tries; i++ {
		sem <- true
		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			defer wg.Done()
			time.Sleep(time.Duration(rand.Int63n(int64(5 * time.Millisecond))))
			cancel()
		}()
		go func(i int) {
			defer wg.Done()
			var dialer Dialer
			// Try to connect to a real host on a port
			// that it is not listening on.
			_, err := dialer.DialContext(ctx, "tcp", "golang.org:3")
			if err == nil {
				t.Errorf("Dial to unbound port succeeded on attempt %d", i)
			}
			<-sem
		}(i)
	}
	wg.Wait()
}
