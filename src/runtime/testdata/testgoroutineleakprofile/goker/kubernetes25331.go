// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: kubernetes
 * Issue or PR  : https://github.com/kubernetes/kubernetes/pull/25331
 * Buggy version: 5dd087040bb13434f1ddf2f0693d0203c30f28cb
 * fix commit-id: 97f4647dc3d8cf46c2b66b89a31c758a6edfb57c
 * Flaky: 100/100
 */
package main

import (
	"context"
	"errors"
	"os"
	"runtime"
	"runtime/pprof"
)

func init() {
	register("Kubernetes25331", Kubernetes25331)
}

type watchChan_kubernetes25331 struct {
	ctx        context.Context
	cancel     context.CancelFunc
	resultChan chan bool
	errChan    chan error
}

func (wc *watchChan_kubernetes25331) Stop() {
	wc.errChan <- errors.New("Error")
	wc.cancel()
}

func (wc *watchChan_kubernetes25331) run() {
	select {
	case err := <-wc.errChan:
		errResult := len(err.Error()) != 0
		wc.cancel() // Removed in fix
		wc.resultChan <- errResult
	case <-wc.ctx.Done():
	}
}

func NewWatchChan_kubernetes25331() *watchChan_kubernetes25331 {
	ctx, cancel := context.WithCancel(context.Background())
	return &watchChan_kubernetes25331{
		ctx:        ctx,
		cancel:     cancel,
		resultChan: make(chan bool),
		errChan:    make(chan error),
	}
}

func Kubernetes25331() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()
	go func() {
		wc := NewWatchChan_kubernetes25331()
		go wc.run()  // G1
		go wc.Stop() // G2
	}()
}
