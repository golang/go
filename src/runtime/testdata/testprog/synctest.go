// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/synctest"
	"sync"
)

func init() {
	register("SynctestCond/signal/no_bubble", func() {
		synctestCond(func(cond *sync.Cond) {
			cond.Signal()
		})
	})
	register("SynctestCond/broadcast/no_bubble", func() {
		synctestCond(func(cond *sync.Cond) {
			cond.Broadcast()
		})
	})
	register("SynctestCond/signal/other_bubble", func() {
		synctestCond(func(cond *sync.Cond) {
			synctest.Run(cond.Signal)
		})
	})
	register("SynctestCond/broadcast/other_bubble", func() {
		synctestCond(func(cond *sync.Cond) {
			synctest.Run(cond.Broadcast)
		})
	})
}

func synctestCond(f func(*sync.Cond)) {
	var (
		mu     sync.Mutex
		cond   = sync.NewCond(&mu)
		readyc = make(chan struct{})
		wg     sync.WaitGroup
	)
	defer wg.Wait()
	wg.Go(func() {
		synctest.Run(func() {
			go func() {
				mu.Lock()
				defer mu.Unlock()
				cond.Wait()
			}()
			synctest.Wait()
			<-readyc // #1: signal that cond.Wait is waiting
			<-readyc // #2: wait to continue
			cond.Signal()
		})
	})
	readyc <- struct{}{}
	f(cond)
}
