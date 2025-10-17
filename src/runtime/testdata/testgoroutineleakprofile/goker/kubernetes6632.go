// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: kubernetes
 * Issue or PR  : https://github.com/kubernetes/kubernetes/pull/6632
 * Buggy version: e597b41d939573502c8dda1dde7bf3439325fb5d
 * fix commit-id: 82afb7ab1fe12cf2efceede2322d082eaf5d5adc
 * Flaky: 4/100
 */
package main

import (
	"os"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Kubernetes6632", Kubernetes6632)
}

type Connection_kubernetes6632 struct {
	closeChan chan bool
}

type idleAwareFramer_kubernetes6632 struct {
	resetChan chan bool
	writeLock sync.Mutex
	conn      *Connection_kubernetes6632
}

func (i *idleAwareFramer_kubernetes6632) monitor() {
	var resetChan = i.resetChan
Loop:
	for {
		select {
		case <-i.conn.closeChan:
			i.writeLock.Lock()
			close(resetChan)
			i.resetChan = nil
			i.writeLock.Unlock()
			break Loop
		}
	}
}

func (i *idleAwareFramer_kubernetes6632) WriteFrame() {
	i.writeLock.Lock()
	defer i.writeLock.Unlock()
	if i.resetChan == nil {
		return
	}
	i.resetChan <- true
}

func NewIdleAwareFramer_kubernetes6632() *idleAwareFramer_kubernetes6632 {
	return &idleAwareFramer_kubernetes6632{
		resetChan: make(chan bool),
		conn: &Connection_kubernetes6632{
			closeChan: make(chan bool),
		},
	}
}

func Kubernetes6632() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() {
			i := NewIdleAwareFramer_kubernetes6632()

			go func() { // helper goroutine
				i.conn.closeChan <- true
			}()
			go i.monitor()    // G1
			go i.WriteFrame() // G2
		}()
	}
}
