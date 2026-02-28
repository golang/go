// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime/pprof"
	"time"
)

func init() {
	register("Kubernetes70277", Kubernetes70277)
}

type WaitFunc_kubernetes70277 func(done <-chan struct{}) <-chan struct{}

type ConditionFunc_kubernetes70277 func() (done bool, err error)

func WaitFor_kubernetes70277(wait WaitFunc_kubernetes70277, fn ConditionFunc_kubernetes70277, done <-chan struct{}) error {
	c := wait(done)
	for {
		_, open := <-c
		ok, err := fn()
		if err != nil {
			return err
		}
		if ok {
			return nil
		}
		if !open {
			break
		}
	}
	return nil
}

func poller_kubernetes70277(interval, timeout time.Duration) WaitFunc_kubernetes70277 {
	return WaitFunc_kubernetes70277(func(done <-chan struct{}) <-chan struct{} {
		ch := make(chan struct{})
		go func() {
			defer close(ch)

			tick := time.NewTicker(interval)
			defer tick.Stop()

			var after <-chan time.Time
			if timeout != 0 {
				timer := time.NewTimer(timeout)
				after = timer.C
				defer timer.Stop()
			}
			for {
				select {
				case <-tick.C:
					select {
					case ch <- struct{}{}:
					default:
					}
				case <-after:
					return
				case <-done:
					return
				}
			}
		}()

		return ch
	})
}

func Kubernetes70277() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i < 1000; i++ {
		go func() {
			stopCh := make(chan struct{})
			defer close(stopCh)
			waitFunc := poller_kubernetes70277(time.Millisecond, 80*time.Millisecond)
			var doneCh <-chan struct{}

			WaitFor_kubernetes70277(func(done <-chan struct{}) <-chan struct{} {
				doneCh = done
				return waitFunc(done)
			}, func() (bool, error) {
				time.Sleep(10 * time.Millisecond)
				return true, nil
			}, stopCh)

			<-doneCh // block here
		}()
	}
}
