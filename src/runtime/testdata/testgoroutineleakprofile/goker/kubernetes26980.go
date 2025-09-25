// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Kubernetes26980", Kubernetes26980)
}

type processorListener_kubernetes26980 struct {
	lock sync.RWMutex
	cond sync.Cond

	pendingNotifications []interface{}
}

func (p *processorListener_kubernetes26980) add(notification interface{}) {
	p.lock.Lock()
	defer p.lock.Unlock()

	p.pendingNotifications = append(p.pendingNotifications, notification)
	p.cond.Broadcast()
}

func (p *processorListener_kubernetes26980) pop(stopCh <-chan struct{}) {
	p.lock.Lock()
	runtime.Gosched()
	defer p.lock.Unlock()
	for {
		for len(p.pendingNotifications) == 0 {
			select {
			case <-stopCh:
				return
			default:
			}
			p.cond.Wait()
		}
		select {
		case <-stopCh:
			return
		}
	}
}

func newProcessListener_kubernetes26980() *processorListener_kubernetes26980 {
	ret := &processorListener_kubernetes26980{
		pendingNotifications: []interface{}{},
	}
	ret.cond.L = &ret.lock
	return ret
}
func Kubernetes26980() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 3000; i++ {
		go func() {
			pl := newProcessListener_kubernetes26980()
			stopCh := make(chan struct{})
			defer close(stopCh)
			pl.add(1)
			runtime.Gosched()
			go pl.pop(stopCh)

			resultCh := make(chan struct{})
			go func() {
				pl.lock.Lock()
				close(resultCh)
			}()
			runtime.Gosched()
			<-resultCh
			pl.lock.Unlock()
		}()
	}
}
