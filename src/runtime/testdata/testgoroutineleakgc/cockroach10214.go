/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/10214
 * Buggy version: 7207111aa3a43df0552509365fdec741a53f873f
 * fix commit-id: 27e863d90ab0660494778f1c35966cc5ddc38e32
 * Flaky: 3/100
 * Description: This deadlock is caused by different order when acquiring
 * coalescedMu.Lock() and raftMu.Lock(). The fix is to refactor sendQueuedHeartbeats()
 * so that cockroachdb can unlock coalescedMu before locking raftMu.
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
	register("Cockroach10214", Cockroach10214)
}

type Store_cockroach10214 struct {
	coalescedMu struct {
		sync.Mutex
		heartbeatResponses []int
	}
	mu struct {
		replicas map[int]*Replica_cockroach10214
	}
}

func (s *Store_cockroach10214) sendQueuedHeartbeats() {
	s.coalescedMu.Lock() // LockA acquire
	runtime.Gosched()
	defer s.coalescedMu.Unlock()
	for i := 0; i < len(s.coalescedMu.heartbeatResponses); i++ {
		s.sendQueuedHeartbeatsToNode() // LockB
	}
	// LockA release
}

func (s *Store_cockroach10214) sendQueuedHeartbeatsToNode() {
	for i := 0; i < len(s.mu.replicas); i++ {
		r := s.mu.replicas[i]
		r.reportUnreachable() // LockB
	}
}

type Replica_cockroach10214 struct {
	raftMu sync.Mutex
	mu     sync.Mutex
	store  *Store_cockroach10214
}

func (r *Replica_cockroach10214) reportUnreachable() {
	r.raftMu.Lock() // LockB acquire
	runtime.Gosched()
	//+time.Sleep(time.Nanosecond)
	defer r.raftMu.Unlock()
	// LockB release
}

func (r *Replica_cockroach10214) tick() {
	r.raftMu.Lock() // LockB acquire
	runtime.Gosched()
	defer r.raftMu.Unlock()
	r.tickRaftMuLocked()
	// LockB release
}

func (r *Replica_cockroach10214) tickRaftMuLocked() {
	r.mu.Lock()
	defer r.mu.Unlock()
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
		r.store.coalescedMu.Lock() // LockA acquire
	default:
		return false
	}
	r.store.coalescedMu.Unlock() // LockA release
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

			rp1 := &Replica_cockroach10214{
				store: store,
			}
			rp2 := &Replica_cockroach10214{
				store: store,
			}
			store.mu.replicas[0] = rp1
			store.mu.replicas[1] = rp2

			go func() {
				// deadlocks: x > 0
				store.sendQueuedHeartbeats()
			}()

			go func() {
				// deadlocks: x > 0
				rp1.tick()
			}()

		}()
	}
}
