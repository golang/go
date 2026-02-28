// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: kubernetes
 * Issue or PR  : https://github.com/kubernetes/kubernetes/pull/1321
 * Buggy version: 9cd0fc70f1ca852c903b18b0933991036b3b2fa1
 * fix commit-id: 435e0b73bb99862f9dedf56a50260ff3dfef14ff
 * Flaky: 1/100
 */
package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Kubernetes1321", Kubernetes1321)
}

type muxWatcher_kubernetes1321 struct {
	result chan struct{}
	m      *Mux_kubernetes1321
	id     int64
}

func (mw *muxWatcher_kubernetes1321) Stop() {
	mw.m.stopWatching(mw.id)
}

type Mux_kubernetes1321 struct {
	lock     sync.Mutex
	watchers map[int64]*muxWatcher_kubernetes1321
}

func NewMux_kubernetes1321() *Mux_kubernetes1321 {
	m := &Mux_kubernetes1321{
		watchers: map[int64]*muxWatcher_kubernetes1321{},
	}
	go m.loop() // G2
	return m
}

func (m *Mux_kubernetes1321) Watch() *muxWatcher_kubernetes1321 {
	mw := &muxWatcher_kubernetes1321{
		result: make(chan struct{}),
		m:      m,
		id:     int64(len(m.watchers)),
	}
	m.watchers[mw.id] = mw
	runtime.Gosched()
	return mw
}

func (m *Mux_kubernetes1321) loop() {
	for i := 0; i < 100; i++ {
		m.distribute()
	}
}

func (m *Mux_kubernetes1321) distribute() {
	m.lock.Lock()
	defer m.lock.Unlock()
	for _, w := range m.watchers {
		w.result <- struct{}{}
		runtime.Gosched()
	}
}

func (m *Mux_kubernetes1321) stopWatching(id int64) {
	m.lock.Lock()
	defer m.lock.Unlock()
	w, ok := m.watchers[id]
	if !ok {
		return
	}
	delete(m.watchers, id)
	close(w.result)
}

func testMuxWatcherClose_kubernetes1321() {
	m := NewMux_kubernetes1321()
	m.watchers[m.Watch().id].Stop()
}

func Kubernetes1321() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i < 1000; i++ {
		go testMuxWatcherClose_kubernetes1321() // G1
	}
}
