// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
)

func init() {
	register("Kubernetes38669", Kubernetes38669)
}

type Event_kubernetes38669 int
type watchCacheEvent_kubernetes38669 int

type cacheWatcher_kubernetes38669 struct {
	sync.Mutex
	input   chan watchCacheEvent_kubernetes38669
	result  chan Event_kubernetes38669
	stopped bool
}

func (c *cacheWatcher_kubernetes38669) process(initEvents []watchCacheEvent_kubernetes38669) {
	for _, event := range initEvents {
		c.sendWatchCacheEvent(&event)
	}
	defer close(c.result)
	defer c.Stop()
	for {
		_, ok := <-c.input
		if !ok {
			return
		}
	}
}

func (c *cacheWatcher_kubernetes38669) sendWatchCacheEvent(event *watchCacheEvent_kubernetes38669) {
	c.result <- Event_kubernetes38669(*event)
}

func (c *cacheWatcher_kubernetes38669) Stop() {
	c.stop()
}

func (c *cacheWatcher_kubernetes38669) stop() {
	c.Lock()
	defer c.Unlock()
	if !c.stopped {
		c.stopped = true
		close(c.input)
	}
}

func newCacheWatcher_kubernetes38669(chanSize int, initEvents []watchCacheEvent_kubernetes38669) *cacheWatcher_kubernetes38669 {
	watcher := &cacheWatcher_kubernetes38669{
		input:   make(chan watchCacheEvent_kubernetes38669, chanSize),
		result:  make(chan Event_kubernetes38669, chanSize),
		stopped: false,
	}
	go watcher.process(initEvents)
	return watcher
}

func Kubernetes38669() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()
	go func() {
		initEvents := []watchCacheEvent_kubernetes38669{1, 2}
		w := newCacheWatcher_kubernetes38669(0, initEvents)
		w.Stop()
	}()
}
