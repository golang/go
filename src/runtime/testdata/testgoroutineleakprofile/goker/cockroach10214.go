// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/10214
 * Buggy version: 7207111aa3a43df0552509365fdec741a53f873f
 * fix commit-id: 27e863d90ab0660494778f1c35966cc5ddc38e32
 * Flaky: 3/100
 * Description: This goroutine leak is caused by different order when acquiring
 * coalescedMu.Lock() and raftMu.Lock(). The fix is to refactor sendQueuedHeartbeats()
 * so that cockroachdb can unlock coalescedMu before locking raftMu.
 */
package main

import (
	"os"
	"runtime/pprof"
	"sync"
	"time"
	"unsafe"
)

func init() {
	register("Cockroach10214", Cockroach10214)
}

type Store_cockroach10214 struct {
	coalescedMu struct {
		sync.Mutex // L1
		heartbeatResponses []int
	}
	mu struct {
		replicas map[int]*Replica_cockroach10214
	}
}

func (s *Store_cockroach10214) sendQueuedHeartbeats() {
	s.coalescedMu.Lock() // L1 acquire
	defer s.coalescedMu.Unlock() // L2 release
	for i := 0; i < len(s.coalescedMu.heartbeatResponses); i++ {
		s.sendQueuedHeartbeatsToNode() // L2
	}
}

func (s *Store_cockroach10214) sendQueuedHeartbeatsToNode() {
	for i := 0; i < len(s.mu.replicas); i++ {
		r := s.mu.replicas[i]
		r.reportUnreachable() // L2
	}
}

type Replica_cockroach10214 struct {
	raftMu sync.Mutex // L2
	mu     sync.Mutex // L3
	store  *Store_cockroach10214
}

func (r *Replica_cockroach10214) reportUnreachable() {
	r.raftMu.Lock() // L2 acquire
	time.Sleep(time.Millisecond)
	defer r.raftMu.Unlock() // L2 release
}

func (r *Replica_cockroach10214) tick() {
	r.raftMu.Lock() // L2 acquire
	defer r.raftMu.Unlock() // L2 release
	r.tickRaftMuLocked()
}

func (r *Replica_cockroach10214) tickRaftMuLocked() {
	r.mu.Lock() // L3 acquire
	defer r.mu.Unlock() // L3 release
	if r.maybeQuiesceLocked() {
		return
	}
}

func (r *Replica_cockroach10214) maybeQuiesceLocked() bool {
	for i := 0; i < 2; i++ {
		if !r.maybeCoalesceHeartbeat() {
			return true
		}
	}
	return false
}

func (r *Replica_cockroach10214) maybeCoalesceHeartbeat() bool {
	msgtype := uintptr(unsafe.Pointer(r)) % 3
	switch msgtype {
	case 0, 1, 2:
		r.store.coalescedMu.Lock() // L1 acquire
	default:
		return false
	}
	r.store.coalescedMu.Unlock() // L1 release
	return true
}

func Cockroach10214() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i < 1000; i++ {
		go func() {
			store := &Store_cockroach10214{}
			responses := &store.coalescedMu.heartbeatResponses
			*responses = append(*responses, 1, 2)
			store.mu.replicas = make(map[int]*Replica_cockroach10214)

			rp1 := &Replica_cockroach10214{ // L2,3[0]
				store: store,
			}
			rp2 := &Replica_cockroach10214{ // L2,3[1]
				store: store,
			}
			store.mu.replicas[0] = rp1
			store.mu.replicas[1] = rp2

			go store.sendQueuedHeartbeats() // G1
			go rp1.tick()                   // G2
		}()
	}
}

// Example of goroutine leak trace:
//
// G1                                      G2
//------------------------------------------------------------------------------------
// s.sendQueuedHeartbeats()                .
// s.coalescedMu.Lock() [L1]               .
// s.sendQueuedHeartbeatsToNode()          .
// s.mu.replicas[0].reportUnreachable()    .
// s.mu.replicas[0].raftMu.Lock() [L2]     .
// .                                       s.mu.replicas[0].tick()
// .                                       s.mu.replicas[0].raftMu.Lock() [L2]
// .                                       s.mu.replicas[0].tickRaftMuLocked()
// .                                       s.mu.replicas[0].mu.Lock() [L3]
// .                                       s.mu.replicas[0].maybeQuiesceLocked()
// .                                       s.mu.replicas[0].maybeCoalesceHeartbeat()
// .                                       s.coalescedMu.Lock() [L1]
//--------------------------------G1,G2 leak------------------------------------------