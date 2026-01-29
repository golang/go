// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/6181
 * Buggy version: c0a232b5521565904b851699853bdbd0c670cf1e
 * fix commit-id: d5814e4886a776bf7789b3c51b31f5206480d184
 * Flaky: 57/100
 */
package main

import (
	"io"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Cockroach6181", Cockroach6181)
}

type testDescriptorDB_cockroach6181 struct {
	cache *rangeDescriptorCache_cockroach6181
}

func initTestDescriptorDB_cockroach6181() *testDescriptorDB_cockroach6181 {
	return &testDescriptorDB_cockroach6181{&rangeDescriptorCache_cockroach6181{}}
}

type rangeDescriptorCache_cockroach6181 struct {
	rangeCacheMu sync.RWMutex
}

func (rdc *rangeDescriptorCache_cockroach6181) LookupRangeDescriptor() {
	rdc.rangeCacheMu.RLock()
	runtime.Gosched()
	io.Discard.Write([]byte(rdc.String()))
	rdc.rangeCacheMu.RUnlock()
	rdc.rangeCacheMu.Lock()
	rdc.rangeCacheMu.Unlock()
}

func (rdc *rangeDescriptorCache_cockroach6181) String() string {
	rdc.rangeCacheMu.RLock()
	defer rdc.rangeCacheMu.RUnlock()
	return rdc.stringLocked()
}

func (rdc *rangeDescriptorCache_cockroach6181) stringLocked() string {
	return "something here"
}

func doLookupWithToken_cockroach6181(rc *rangeDescriptorCache_cockroach6181) {
	rc.LookupRangeDescriptor()
}

func testRangeCacheCoalescedRequests_cockroach6181() {
	db := initTestDescriptorDB_cockroach6181()
	pauseLookupResumeAndAssert := func() {
		var wg sync.WaitGroup
		for i := 0; i < 3; i++ {
			wg.Add(1)
			go func() { // G2,G3,...
				doLookupWithToken_cockroach6181(db.cache)
				wg.Done()
			}()
		}
		wg.Wait()
	}
	pauseLookupResumeAndAssert()
}

func Cockroach6181() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i < 100; i++ {
		go testRangeCacheCoalescedRequests_cockroach6181() // G1
	}
}
