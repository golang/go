package main

import (
	"context"
	"runtime"
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
	le.mu.Lock()
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
	defer func() {
		time.Sleep(10 * time.Millisecond)
		runtime.GC()
	}()

	for i := 0; i < 100; i++ {
		go func() {
			// deadlocks: x > 0

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
}
