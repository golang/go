/*
 * Project: kubernetes
 * Issue or PR  : https://github.com/kubernetes/kubernetes/pull/1321
 * Buggy version: 9cd0fc70f1ca852c903b18b0933991036b3b2fa1
 * fix commit-id: 435e0b73bb99862f9dedf56a50260ff3dfef14ff
 * Flaky: 1/100
 * Description:
 *   This is a lock-channel bug. The first goroutine invokes
 * distribute() function. distribute() function holds m.lock.Lock(),
 * while blocking at sending message to w.result. The second goroutine
 * invokes stopWatching() funciton, which can unblock the first
 * goroutine by closing w.result. However, in order to close w.result,
 * stopWatching() function needs to acquire m.lock.Lock() firstly.
 *   The fix is to introduce another channel and put receive message
 * from the second channel in the same select as the w.result. Close
 * the second channel can unblock the first goroutine, while no need
 * to hold m.lock.Lock().
 */
package main

import (
	"runtime"
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
	// deadlocks: x > 0
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
	// deadlocks: x > 0
	m := NewMux_kubernetes1321()
	m.watchers[m.Watch().id].Stop()
}

///
/// G1 							G2
/// testMuxWatcherClose()
/// NewMux()
/// 							m.loop()
/// 							m.distribute()
/// 							m.lock.Lock()
/// 							w.result <- true
/// w := m.Watch()
/// w.Stop()
/// mw.m.stopWatching()
/// m.lock.Lock()
/// ---------------G1,G2 deadlock---------------
///

func Kubernetes1321() {
	defer func() {
		time.Sleep(100 * time.Millisecond)
		runtime.GC()
	}()
	for i := 0; i < 1000; i++ {
		go testMuxWatcherClose_kubernetes1321() // G1
	}
}
