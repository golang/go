// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: moby
 * Issue or PR  : https://github.com/moby/moby/pull/33781
 * Buggy version: 33fd3817b0f5ca4b87f0a75c2bd583b4425d392b
 * fix commit-id: 67297ba0051d39be544009ba76abea14bc0be8a4
 * Flaky: 25/100
 */

package main

import (
	"context"
	"os"
	"runtime/pprof"
	"time"
)

func init() {
	register("Moby33781", Moby33781)
}

func monitor_moby33781(stop chan bool) {
	probeInterval := time.Millisecond
	probeTimeout := time.Millisecond
	for {
		select {
		case <-stop:
			return
		case <-time.After(probeInterval):
			results := make(chan bool)
			ctx, cancelProbe := context.WithTimeout(context.Background(), probeTimeout)
			go func() { // G3
				results <- true
				close(results)
			}()
			select {
			case <-stop:
				// results should be drained here
				cancelProbe()
				return
			case <-results:
				cancelProbe()
			case <-ctx.Done():
				cancelProbe()
				<-results
			}
		}
	}
}

func Moby33781() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i < 100; i++ {
		go func(i int) {
			stop := make(chan bool)
			go monitor_moby33781(stop) // G1
			go func() {                // G2
				time.Sleep(time.Duration(i) * time.Millisecond)
				stop <- true
			}()
		}(i)
	}
}
