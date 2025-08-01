package main

import (
	"runtime"
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
	defer func() {
		time.Sleep(100 * time.Millisecond)
		runtime.GC()
	}()

	for i := 0; i < 3000; i++ {
		go func() {
			// deadlocks: x > 0
			pl := newProcessListener_kubernetes26980()
			stopCh := make(chan struct{})
			defer close(stopCh)
			pl.add(1)
			runtime.Gosched()
			// deadlocks: x > 0
			go pl.pop(stopCh)

			resultCh := make(chan struct{})
			go func() {
				// deadlocks: x > 0
				pl.lock.Lock()
				close(resultCh)
			}()
			runtime.Gosched()
			<-resultCh
			pl.lock.Unlock()
		}()
	}
}
