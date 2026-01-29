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
	register("Syncthing5795", Syncthing5795)
}

type message_syncthing5795 interface{}

type ClusterConfig_syncthing5795 struct{}

type Model_syncthing5795 interface {
	ClusterConfig(message_syncthing5795)
}

type TestModel_syncthing5795 struct {
	ccFn func()
}

func (t *TestModel_syncthing5795) ClusterConfig(msg message_syncthing5795) {
	if t.ccFn != nil {
		t.ccFn()
	}
}

type Connection_syncthing5795 interface {
	Start()
	Close()
}

type rawConnection_syncthing5795 struct {
	receiver Model_syncthing5795

	inbox                 chan message_syncthing5795
	dispatcherLoopStopped chan struct{}
	closed                chan struct{}
	closeOnce             sync.Once
}

func (c *rawConnection_syncthing5795) Start() {
	go c.dispatcherLoop() // G2
}

func (c *rawConnection_syncthing5795) dispatcherLoop() {
	defer close(c.dispatcherLoopStopped)
	var msg message_syncthing5795
	for {
		select {
		case msg = <-c.inbox:
		case <-c.closed:
			return
		}
		switch msg := msg.(type) {
		case *ClusterConfig_syncthing5795:
			c.receiver.ClusterConfig(msg)
		default:
			return
		}
	}
}

func (c *rawConnection_syncthing5795) Close() {
	c.closeOnce.Do(func() {
		close(c.closed)
		<-c.dispatcherLoopStopped
	})
}

func Syncthing5795() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()
	go func() { // G1
		m := &TestModel_syncthing5795{}
		c := &rawConnection_syncthing5795{
			dispatcherLoopStopped: make(chan struct{}),
			closed:                make(chan struct{}),
			inbox:                 make(chan message_syncthing5795),
			receiver:              m,
		}
		m.ccFn = c.Close

		c.Start()
		c.inbox <- &ClusterConfig_syncthing5795{}

		<-c.dispatcherLoopStopped
	}()
}
