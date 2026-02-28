// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: grpc-go
 * Issue or PR  : https://github.com/grpc/grpc-go/pull/660
 * Buggy version: db85417dd0de6cc6f583672c6175a7237e5b5dd2
 * fix commit-id: ceacfbcbc1514e4e677932fd55938ac455d182fb
 * Flaky: 100/100
 */
package main

import (
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
)

func init() {
	register("Grpc660", Grpc660)
}

type benchmarkClient_grpc660 struct {
	stop chan bool
}

func (bc *benchmarkClient_grpc660) doCloseLoopUnary() {
	for {
		done := make(chan bool)
		go func() { // G2
			if rand.Intn(10) > 7 {
				done <- false
				return
			}
			done <- true
		}()
		select {
		case <-bc.stop:
			return
		case <-done:
		}
	}
}

func Grpc660() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()
	go func() {
		bc := &benchmarkClient_grpc660{
			stop: make(chan bool),
		}
		go bc.doCloseLoopUnary() // G1
		go func() {              // helper goroutine
			bc.stop <- true
		}()
	}()
}
