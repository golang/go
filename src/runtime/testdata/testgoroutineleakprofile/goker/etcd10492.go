// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Etcd10492", Etcd10492)
}

type Checkpointer_etcd10492 func(ctx context.Context)

type lessor_etcd10492 struct {
	mu                 sync.RWMutex
	cp                 Checkpointer_etcd10492
	checkpointInterval time.Duration
}

func (le *lessor_etcd10492) Checkpoint() {
	le.mu.Lock() // Lock acquired twice here
	defer le.mu.Unlock()
}

func (le *lessor_etcd10492) SetCheckpointer(cp Checkpointer_etcd10492) {
	le.mu.Lock()
	defer le.mu.Unlock()

	le.cp = cp
}

func (le *lessor_etcd10492) Renew() {
	le.mu.Lock()
	unlock := func() { le.mu.Unlock() }
	defer func() { unlock() }()

	if le.cp != nil {
		le.cp(context.Background())
	}
}

func Etcd10492() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()

	go func() { // G1
		le := &lessor_etcd10492{
			checkpointInterval: 0,
		}
		fakerCheckerpointer_etcd10492 := func(ctx context.Context) {
			le.Checkpoint()
		}
		le.SetCheckpointer(fakerCheckerpointer_etcd10492)
		le.mu.Lock()
		le.mu.Unlock()
		le.Renew()
	}()
}
