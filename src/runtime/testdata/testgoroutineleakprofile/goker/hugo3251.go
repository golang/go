// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Hugo3251", Hugo3251)
}

type remoteLock_hugo3251 struct {
	sync.RWMutex                        // L1
	m            map[string]*sync.Mutex // L2
}

func (l *remoteLock_hugo3251) URLLock(url string) {
	l.Lock() // L1
	if _, ok := l.m[url]; !ok {
		l.m[url] = &sync.Mutex{}
	}
	l.m[url].Lock() // L2
	runtime.Gosched()
	l.Unlock() // L1
	// runtime.Gosched()
}

func (l *remoteLock_hugo3251) URLUnlock(url string) {
	l.RLock()         // L1
	defer l.RUnlock() // L1
	if um, ok := l.m[url]; ok {
		um.Unlock() // L2
	}
}

func resGetRemote_hugo3251(remoteURLLock *remoteLock_hugo3251, url string) error {
	remoteURLLock.URLLock(url)
	defer func() { remoteURLLock.URLUnlock(url) }()

	return nil
}

func Hugo3251() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(time.Second)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 11; i++ {
		go func() { // G1
			url := "http://Foo.Bar/foo_Bar-Foo"
			remoteURLLock := &remoteLock_hugo3251{m: make(map[string]*sync.Mutex)}
			for range []bool{false, true} {
				var wg sync.WaitGroup
				for i := 0; i < 100; i++ {
					wg.Add(1)
					go func(gor int) { // G2
						defer wg.Done()
						for j := 0; j < 200; j++ {
							err := resGetRemote_hugo3251(remoteURLLock, url)
							if err != nil {
								fmt.Errorf("Error getting resource content: %s", err)
							}
							time.Sleep(300 * time.Nanosecond)
						}
					}(i)
				}
				wg.Wait()
			}
		}()
	}
}