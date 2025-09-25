// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Kubernetes11298", Kubernetes11298)
}

type Signal_kubernetes11298 <-chan struct{}

func After_kubernetes11298(f func()) Signal_kubernetes11298 {
	ch := make(chan struct{})
	go func() {
		defer close(ch)
		if f != nil {
			f()
		}
	}()
	return Signal_kubernetes11298(ch)
}

func Until_kubernetes11298(f func(), period time.Duration, stopCh <-chan struct{}) {
	if f == nil {
		return
	}
	for {
		select {
		case <-stopCh:
			return
		default:
		}
		f()
		select {
		case <-stopCh:
		case <-time.After(period):
		}
	}

}

type notifier_kubernetes11298 struct {
	lock sync.Mutex
	cond *sync.Cond
}

// abort will be closed no matter what
func (n *notifier_kubernetes11298) serviceLoop(abort <-chan struct{}) {
	n.lock.Lock()
	defer n.lock.Unlock()
	for {
		select {
		case <-abort:
			return
		default:
			ch := After_kubernetes11298(func() {
				n.cond.Wait()
			})
			select {
			case <-abort:
				n.cond.Signal()
				<-ch
				return
			case <-ch:
			}
		}
	}
}

// abort will be closed no matter what
func Notify_kubernetes11298(abort <-chan struct{}) {
	n := &notifier_kubernetes11298{}
	n.cond = sync.NewCond(&n.lock)
	finished := After_kubernetes11298(func() {
		Until_kubernetes11298(func() {
			for {
				select {
				case <-abort:
					return
				default:
					func() {
						n.lock.Lock()
						defer n.lock.Unlock()
						n.cond.Signal()
					}()
				}
			}
		}, 0, abort)
	})
	Until_kubernetes11298(func() { n.serviceLoop(finished) }, 0, abort)
}
func Kubernetes11298() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 1000; i++ {
		go func() {
			done := make(chan struct{})
			notifyDone := After_kubernetes11298(func() { Notify_kubernetes11298(done) })
			go func() {
				defer close(done)
				time.Sleep(300 * time.Nanosecond)
			}()
			<-notifyDone
		}()
	}
}
