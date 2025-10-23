// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: kubernetes
 * Tag: Reproduce misbehavior
 * Issue or PR  : https://github.com/kubernetes/kubernetes/pull/58107
 * Buggy version: 2f17d782eb2772d6401da7ddced9ac90656a7a79
 * fix commit-id: 010a127314a935d8d038f8dd4559fc5b249813e4
 * Flaky: 53/100
 */

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Kubernetes58107", Kubernetes58107)
}

type RateLimitingInterface_kubernetes58107 interface {
	Get()
	Put()
}

type Type_kubernetes58107 struct {
	cond *sync.Cond
}

func (q *Type_kubernetes58107) Get() {
	q.cond.L.Lock()
	defer q.cond.L.Unlock()
	q.cond.Wait()
}

func (q *Type_kubernetes58107) Put() {
	q.cond.Signal()
}

type ResourceQuotaController_kubernetes58107 struct {
	workerLock        sync.RWMutex
	queue             RateLimitingInterface_kubernetes58107
	missingUsageQueue RateLimitingInterface_kubernetes58107
}

func (rq *ResourceQuotaController_kubernetes58107) worker(queue RateLimitingInterface_kubernetes58107, _ string) {
	workFunc := func() bool {
		rq.workerLock.RLock()
		defer rq.workerLock.RUnlock()
		queue.Get()
		return true
	}
	for {
		if quit := workFunc(); quit {
			return
		}
	}
}

func (rq *ResourceQuotaController_kubernetes58107) Run() {
	go rq.worker(rq.queue, "G1")             // G3
	go rq.worker(rq.missingUsageQueue, "G2") // G4
}

func (rq *ResourceQuotaController_kubernetes58107) Sync() {
	for i := 0; i < 100000; i++ {
		rq.workerLock.Lock()
		runtime.Gosched()
		rq.workerLock.Unlock()
	}
}

func (rq *ResourceQuotaController_kubernetes58107) HelperSignals() {
	for i := 0; i < 100000; i++ {
		rq.queue.Put()
		rq.missingUsageQueue.Put()
	}
}

func startResourceQuotaController_kubernetes58107() {
	resourceQuotaController := &ResourceQuotaController_kubernetes58107{
		queue:             &Type_kubernetes58107{sync.NewCond(&sync.Mutex{})},
		missingUsageQueue: &Type_kubernetes58107{sync.NewCond(&sync.Mutex{})},
	}

	go resourceQuotaController.Run()  // G2
	go resourceQuotaController.Sync() // G5
	resourceQuotaController.HelperSignals()
}

func Kubernetes58107() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(1000 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 1000; i++ {
		go startResourceQuotaController_kubernetes58107() // G1
	}
}
