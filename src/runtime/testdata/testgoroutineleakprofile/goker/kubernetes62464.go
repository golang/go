// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: kubernetes
 * Issue or PR  : https://github.com/kubernetes/kubernetes/pull/62464
 * Buggy version: a048ca888ad27367b1a7b7377c67658920adbf5d
 * fix commit-id: c1b19fce903675b82e9fdd1befcc5f5d658bfe78
 * Flaky: 8/100
 */

package main

import (
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Kubernetes62464", Kubernetes62464)
}

type State_kubernetes62464 interface {
	GetCPUSetOrDefault()
	GetCPUSet() bool
	GetDefaultCPUSet()
	SetDefaultCPUSet()
}

type stateMemory_kubernetes62464 struct {
	sync.RWMutex
}

func (s *stateMemory_kubernetes62464) GetCPUSetOrDefault() {
	s.RLock()
	defer s.RUnlock()
	if ok := s.GetCPUSet(); ok {
		return
	}
	s.GetDefaultCPUSet()
}

func (s *stateMemory_kubernetes62464) GetCPUSet() bool {
	runtime.Gosched()
	s.RLock()
	defer s.RUnlock()

	if rand.Intn(10) > 5 {
		return true
	}
	return false
}

func (s *stateMemory_kubernetes62464) GetDefaultCPUSet() {
	s.RLock()
	defer s.RUnlock()
}

func (s *stateMemory_kubernetes62464) SetDefaultCPUSet() {
	s.Lock()
	runtime.Gosched()
	defer s.Unlock()
}

type staticPolicy_kubernetes62464 struct{}

func (p *staticPolicy_kubernetes62464) RemoveContainer(s State_kubernetes62464) {
	s.GetDefaultCPUSet()
	s.SetDefaultCPUSet()
}

type manager_kubernetes62464 struct {
	state *stateMemory_kubernetes62464
}

func (m *manager_kubernetes62464) reconcileState() {
	m.state.GetCPUSetOrDefault()
}

func NewPolicyAndManager_kubernetes62464() (*staticPolicy_kubernetes62464, *manager_kubernetes62464) {
	s := &stateMemory_kubernetes62464{}
	m := &manager_kubernetes62464{s}
	p := &staticPolicy_kubernetes62464{}
	return p, m
}

///
/// G1 									G2
/// m.reconcileState()
/// m.state.GetCPUSetOrDefault()
/// s.RLock()
/// s.GetCPUSet()
/// 									p.RemoveContainer()
/// 									s.GetDefaultCPUSet()
/// 									s.SetDefaultCPUSet()
/// 									s.Lock()
/// s.RLock()
/// ---------------------G1,G2 deadlock---------------------
///

func Kubernetes62464() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 1000; i++ {
		go func() {
			p, m := NewPolicyAndManager_kubernetes62464()
			go m.reconcileState()
			go p.RemoveContainer(m.state)
		}()
	}
}
