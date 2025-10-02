// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: etcd
 * Issue or PR  : https://github.com/coreos/etcd/pull/7902
 * Buggy version: dfdaf082c51ba14861267f632f6af795a27eb4ef
 * fix commit-id: 87d99fe0387ee1df1cf1811d88d37331939ef4ae
 * Flaky: 100/100
 */
package main

import (
	"os"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Etcd7902", Etcd7902)
}

type roundClient_etcd7902 struct {
	progress int
	acquire  func()
	validate func()
	release  func()
}

func runElectionFunc_etcd7902() {
	rcs := make([]roundClient_etcd7902, 3)
	nextc := make(chan bool)
	for i := range rcs {
		var rcNextc chan bool
		setRcNextc := func() {
			rcNextc = nextc
		}
		rcs[i].acquire = func() {}
		rcs[i].validate = func() {
			setRcNextc()
		}
		rcs[i].release = func() {
			if i == 0 { // Assume the first roundClient is the leader
				close(nextc)
				nextc = make(chan bool)
			}
			<-rcNextc // Follower is blocking here
		}
	}
	doRounds_etcd7902(rcs, 100)
}

func doRounds_etcd7902(rcs []roundClient_etcd7902, rounds int) {
	var mu sync.Mutex
	var wg sync.WaitGroup
	wg.Add(len(rcs))
	for i := range rcs {
		go func(rc *roundClient_etcd7902) { // G2,G3
			defer wg.Done()
			for rc.progress < rounds || rounds <= 0 {
				rc.acquire()
				mu.Lock()
				rc.validate()
				mu.Unlock()
				time.Sleep(10 * time.Millisecond)
				rc.progress++
				mu.Lock()
				rc.release()
				mu.Unlock()
			}
		}(&rcs[i])
	}
	wg.Wait()
}

func Etcd7902() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i < 100; i++ {
		go runElectionFunc_etcd7902() // G1
	}
}
