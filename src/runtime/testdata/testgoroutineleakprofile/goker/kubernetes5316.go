// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: kubernetes
 * Issue or PR  : https://github.com/kubernetes/kubernetes/pull/5316
 * Buggy version: c868b0bbf09128960bc7c4ada1a77347a464d876
 * fix commit-id: cc3a433a7abc89d2f766d4c87eaae9448e3dc091
 * Flaky: 100/100
 */

package main

import (
	"errors"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"time"
)

func init() {
	register("Kubernetes5316", Kubernetes5316)
}

func finishRequest_kubernetes5316(timeout time.Duration, fn func() error) {
	ch := make(chan bool)
	errCh := make(chan error)
	go func() { // G2
		if err := fn(); err != nil {
			errCh <- err
		} else {
			ch <- true
		}
	}()

	select {
	case <-ch:
	case <-errCh:
	case <-time.After(timeout):
	}
}

func Kubernetes5316() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Wait a bit because the child goroutine relies on timed operations.
		time.Sleep(100 * time.Millisecond)

		// Yield several times to allow the child goroutine to run
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()
	go func() {
		fn := func() error {
			time.Sleep(2 * time.Millisecond)
			if rand.Intn(10) > 5 {
				return errors.New("Error")
			}
			return nil
		}
		go finishRequest_kubernetes5316(time.Microsecond, fn) // G1
	}()
}
