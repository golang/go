// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/7504
 * Buggy version: bc963b438cdc3e0ad058a5282358e5aee0595e17
 * fix commit-id: cab761b9f5ee5dee1448bc5d6b1d9f5a0ff0bad5
 * Flaky: 1/100
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
	register("Cockroach7504", Cockroach7504)
}

func MakeCacheKey_cockroach7504(lease *LeaseState_cockroach7504) int {
	return lease.id
}

type LeaseState_cockroach7504 struct {
	mu sync.Mutex // L1
	id int
}
type LeaseSet_cockroach7504 struct {
	data []*LeaseState_cockroach7504
}

func (l *LeaseSet_cockroach7504) find(id int) *LeaseState_cockroach7504 {
	return l.data[id]
}

func (l *LeaseSet_cockroach7504) remove(s *LeaseState_cockroach7504) {
	for i := 0; i < len(l.data); i++ {
		if s == l.data[i] {
			l.data = append(l.data[:i], l.data[i+1:]...)
			break
		}
	}
}

type tableState_cockroach7504 struct {
	tableNameCache *tableNameCache_cockroach7504
	mu             sync.Mutex // L3
	active         *LeaseSet_cockroach7504
}

func (t *tableState_cockroach7504) release(lease *LeaseState_cockroach7504) {
	t.mu.Lock()         // L3
	defer t.mu.Unlock() // L3

	s := t.active.find(MakeCacheKey_cockroach7504(lease))
	s.mu.Lock() // L1
	runtime.Gosched()
	defer s.mu.Unlock() // L1

	t.removeLease(s)
}
func (t *tableState_cockroach7504) removeLease(lease *LeaseState_cockroach7504) {
	t.active.remove(lease)
	t.tableNameCache.remove(lease) // L1 acquire/release
}

type tableNameCache_cockroach7504 struct {
	mu     sync.Mutex // L2
	tables map[int]*LeaseState_cockroach7504
}

func (c *tableNameCache_cockroach7504) get(id int) {
	c.mu.Lock()         // L2
	defer c.mu.Unlock() // L2
	lease, ok := c.tables[id]
	if !ok {
		return
	}
	if lease == nil {
		panic("nil lease in name cache")
	}
	lease.mu.Lock()         // L1
	defer lease.mu.Unlock() // L1
}

func (c *tableNameCache_cockroach7504) remove(lease *LeaseState_cockroach7504) {
	c.mu.Lock() // L2
	runtime.Gosched()
	defer c.mu.Unlock() // L2
	key := MakeCacheKey_cockroach7504(lease)
	existing, ok := c.tables[key]
	if !ok {
		return
	}
	if existing == lease {
		delete(c.tables, key)
	}
}

type LeaseManager_cockroach7504 struct {
	_          [64]byte
	tableNames *tableNameCache_cockroach7504
	tables     map[int]*tableState_cockroach7504
}

func (m *LeaseManager_cockroach7504) AcquireByName(id int) {
	m.tableNames.get(id)
}

func (m *LeaseManager_cockroach7504) findTableState(lease *LeaseState_cockroach7504) *tableState_cockroach7504 {
	existing, ok := m.tables[lease.id]
	if !ok {
		return nil
	}
	return existing
}

func (m *LeaseManager_cockroach7504) Release(lease *LeaseState_cockroach7504) {
	t := m.findTableState(lease)
	t.release(lease)
}
func NewLeaseManager_cockroach7504(tname *tableNameCache_cockroach7504, ts *tableState_cockroach7504) *LeaseManager_cockroach7504 {
	mgr := &LeaseManager_cockroach7504{
		tableNames: tname,
		tables:     make(map[int]*tableState_cockroach7504),
	}
	mgr.tables[0] = ts
	return mgr
}
func NewLeaseSet_cockroach7504(n int) *LeaseSet_cockroach7504 {
	lset := &LeaseSet_cockroach7504{}
	for i := 0; i < n; i++ {
		lease := new(LeaseState_cockroach7504)
		lset.data = append(lset.data, lease)
	}
	return lset
}

func Cockroach7504() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	for i := 0; i < 100; i++ {
		go func() {
			leaseNum := 2
			lset := NewLeaseSet_cockroach7504(leaseNum)

			nc := &tableNameCache_cockroach7504{
				tables: make(map[int]*LeaseState_cockroach7504),
			}
			for i := 0; i < leaseNum; i++ {
				nc.tables[i] = lset.find(i)
			}

			ts := &tableState_cockroach7504{
				tableNameCache: nc,
				active:         lset,
			}

			mgr := NewLeaseManager_cockroach7504(nc, ts)

			// G1
			go func() {
				// lock L2-L1
				mgr.AcquireByName(0)
			}()

			// G2
			go func() {
				// lock L1-L2
				mgr.Release(lset.find(0))
			}()
		}()
	}
}
