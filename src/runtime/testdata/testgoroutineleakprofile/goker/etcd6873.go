// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: etcd
 * Issue or PR  : https://github.com/etcd-io/etcd/commit/7618fdd1d642e47cac70c03f637b0fd798a53a6e
 * Buggy version: 377f19b0031f9c0aafe2aec28b6f9019311f52f9
 * fix commit-id: 7618fdd1d642e47cac70c03f637b0fd798a53a6e
 * Flaky: 9/100
 */
package main

import (
	"os"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Etcd6873", Etcd6873)
}

type watchBroadcast_etcd6873 struct{}

type watchBroadcasts_etcd6873 struct {
	mu      sync.Mutex
	updatec chan *watchBroadcast_etcd6873
	donec   chan struct{}
}

func newWatchBroadcasts_etcd6873() *watchBroadcasts_etcd6873 {
	wbs := &watchBroadcasts_etcd6873{
		updatec: make(chan *watchBroadcast_etcd6873, 1),
		donec:   make(chan struct{}),
	}
	go func() { // G2
		defer close(wbs.donec)
		for wb := range wbs.updatec {
			wbs.coalesce(wb)
		}
	}()
	return wbs
}

func (wbs *watchBroadcasts_etcd6873) coalesce(wb *watchBroadcast_etcd6873) {
	wbs.mu.Lock()
	wbs.mu.Unlock()
}

func (wbs *watchBroadcasts_etcd6873) stop() {
	wbs.mu.Lock()
	defer wbs.mu.Unlock()
	close(wbs.updatec)
	<-wbs.donec
}

func (wbs *watchBroadcasts_etcd6873) update(wb *watchBroadcast_etcd6873) {
	select {
	case wbs.updatec <- wb:
	default:
	}
}

// Example of goroutine leak trace:
//
// G1                   G2                  G3
//---------------------------------------------------------
// newWatchBroadcasts()
//	wbs.update()
// wbs.updatec <-
// return
//                      <-wbs.updatec
//                      wbs.coalesce()
//                                         wbs.stop()
//                                         wbs.mu.Lock()
//                                         close(wbs.updatec)
//                                         <-wbs.donec
//                      wbs.mu.Lock()
//---------------------G2,G3 leak-------------------------
//

func Etcd6873() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() {
			wbs := newWatchBroadcasts_etcd6873() // G1
			wbs.update(&watchBroadcast_etcd6873{})
			go wbs.stop() // G3
		}()
	}
}
