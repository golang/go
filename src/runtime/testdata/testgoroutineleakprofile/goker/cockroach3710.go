// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/3710
 * Buggy version: 4afdd4860fd7c3bd9e92489f84a95e5cc7d11a0d
 * fix commit-id: cb65190f9caaf464723e7d072b1f1b69a044ef7b
 * Flaky: 2/100
 */

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
	"unsafe"
)

func init() {
	register("Cockroach3710", Cockroach3710)
}

type Store_cockroach3710 struct {
	raftLogQueue *baseQueue
	replicas     map[int]*Replica_cockroach3710

	mu struct {
		sync.RWMutex
	}
}

func (s *Store_cockroach3710) ForceRaftLogScanAndProcess() {
	s.mu.RLock()
	runtime.Gosched()
	for _, r := range s.replicas {
		s.raftLogQueue.MaybeAdd(r)
	}
	s.mu.RUnlock()
}

func (s *Store_cockroach3710) RaftStatus() {
	s.mu.RLock()
	defer s.mu.RUnlock()
}

func (s *Store_cockroach3710) processRaft() {
	go func() {
		for {
			var replicas []*Replica_cockroach3710
			s.mu.Lock()
			for _, r := range s.replicas {
				replicas = append(replicas, r)
			}
			s.mu.Unlock()
			break
		}
	}()
}

type Replica_cockroach3710 struct {
	store *Store_cockroach3710
}

type baseQueue struct {
	sync.Mutex
	impl *raftLogQueue
}

func (bq *baseQueue) MaybeAdd(repl *Replica_cockroach3710) {
	bq.Lock()
	defer bq.Unlock()
	bq.impl.shouldQueue(repl)
}

type raftLogQueue struct{}

func (*raftLogQueue) shouldQueue(r *Replica_cockroach3710) {
	getTruncatableIndexes(r)
}

func getTruncatableIndexes(r *Replica_cockroach3710) {
	r.store.RaftStatus()
}

func NewStore_cockroach3710() *Store_cockroach3710 {
	rlq := &raftLogQueue{}
	bq := &baseQueue{impl: rlq}
	store := &Store_cockroach3710{
		raftLogQueue: bq,
		replicas:     make(map[int]*Replica_cockroach3710),
	}
	r1 := &Replica_cockroach3710{store}
	r2 := &Replica_cockroach3710{store}

	makeKey := func(r *Replica_cockroach3710) int {
		return int((uintptr(unsafe.Pointer(r)) >> 1) % 7)
	}
	store.replicas[makeKey(r1)] = r1
	store.replicas[makeKey(r2)] = r2

	return store
}

func Cockroach3710() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i < 10000; i++ {
		go func() {
			store := NewStore_cockroach3710()
			go store.ForceRaftLogScanAndProcess() // G1
			go store.processRaft()                // G2
		}()
	}
}
