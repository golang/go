// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

// This test case is a reproduction of grpc/3017.
//
// It is a goroutine leak that also simultaneously engages many GC assists.
// Testing runtime behaviour when pivoting between regular and goroutine leak detection modes.

func init() {
	register("Grpc3017", Grpc3017)
}

type Address_grpc3017 int
type SubConn_grpc3017 int

type subConnCacheEntry_grpc3017 struct {
	sc            SubConn_grpc3017
	cancel        func()
	abortDeleting bool
}

type lbCacheClientConn_grpc3017 struct {
	mu            sync.Mutex // L1
	timeout       time.Duration
	subConnCache  map[Address_grpc3017]*subConnCacheEntry_grpc3017
	subConnToAddr map[SubConn_grpc3017]Address_grpc3017
}

func (ccc *lbCacheClientConn_grpc3017) NewSubConn(addrs []Address_grpc3017) SubConn_grpc3017 {
	if len(addrs) != 1 {
		return SubConn_grpc3017(1)
	}
	addrWithoutMD := addrs[0]
	ccc.mu.Lock() // L1
	defer ccc.mu.Unlock()
	if entry, ok := ccc.subConnCache[addrWithoutMD]; ok {
		entry.cancel()
		delete(ccc.subConnCache, addrWithoutMD)
		return entry.sc
	}
	scNew := SubConn_grpc3017(1)
	ccc.subConnToAddr[scNew] = addrWithoutMD
	return scNew
}

func (ccc *lbCacheClientConn_grpc3017) RemoveSubConn(sc SubConn_grpc3017) {
	ccc.mu.Lock() // L1
	defer ccc.mu.Unlock()
	addr, ok := ccc.subConnToAddr[sc]
	if !ok {
		return
	}

	if entry, ok := ccc.subConnCache[addr]; ok {
		if entry.sc != sc {
			delete(ccc.subConnToAddr, sc)
		}
		return
	}

	entry := &subConnCacheEntry_grpc3017{
		sc: sc,
	}
	ccc.subConnCache[addr] = entry

	timer := time.AfterFunc(ccc.timeout, func() { // G3
		runtime.Gosched()
		ccc.mu.Lock() // L1
		if entry.abortDeleting {
			return // Missing unlock
		}
		delete(ccc.subConnToAddr, sc)
		delete(ccc.subConnCache, addr)
		ccc.mu.Unlock()
	})

	entry.cancel = func() {
		if !timer.Stop() {
			entry.abortDeleting = true
		}
	}
}

func Grpc3017() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() { //G1
			done := make(chan struct{})

			ccc := &lbCacheClientConn_grpc3017{
				timeout:       time.Nanosecond,
				subConnCache:  make(map[Address_grpc3017]*subConnCacheEntry_grpc3017),
				subConnToAddr: make(map[SubConn_grpc3017]Address_grpc3017),
			}

			sc := ccc.NewSubConn([]Address_grpc3017{Address_grpc3017(1)})
			go func() { // G2
				for i := 0; i < 10000; i++ {
					ccc.RemoveSubConn(sc)
					sc = ccc.NewSubConn([]Address_grpc3017{Address_grpc3017(1)})
				}
				close(done)
			}()
			<-done
		}()
	}
}
