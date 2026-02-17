// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: kubernetes
 * Issue or PR  : https://github.com/kubernetes/kubernetes/pull/10182
 * Buggy version: 4b990d128a17eea9058d28a3b3688ab8abafbd94
 * fix commit-id: 64ad3e17ad15cd0f9a4fd86706eec1c572033254
 * Flaky: 15/100
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
	register("Kubernetes10182", Kubernetes10182)
}

type statusManager_kubernetes10182 struct {
	podStatusesLock  sync.RWMutex
	podStatusChannel chan bool
}

func (s *statusManager_kubernetes10182) Start() {
	go func() {
		for i := 0; i < 2; i++ {
			s.syncBatch()
		}
	}()
}

func (s *statusManager_kubernetes10182) syncBatch() {
	runtime.Gosched()
	<-s.podStatusChannel
	s.DeletePodStatus()
}

func (s *statusManager_kubernetes10182) DeletePodStatus() {
	s.podStatusesLock.Lock()
	defer s.podStatusesLock.Unlock()
}

func (s *statusManager_kubernetes10182) SetPodStatus() {
	s.podStatusesLock.Lock()
	defer s.podStatusesLock.Unlock()
	s.podStatusChannel <- true
}

func NewStatusManager_kubernetes10182() *statusManager_kubernetes10182 {
	return &statusManager_kubernetes10182{
		podStatusChannel: make(chan bool),
	}
}

// 	Example of deadlock trace:
//
//	G1 						G2							G3
//	--------------------------------------------------------------------------------
//	s.Start()
//	s.syncBatch()
//							s.SetPodStatus()
//	<-s.podStatusChannel
//							s.podStatusesLock.Lock()
//							s.podStatusChannel <- true
//							s.podStatusesLock.Unlock()
//							return
//	s.DeletePodStatus()
//														s.podStatusesLock.Lock()
//														s.podStatusChannel <- true
//	s.podStatusesLock.Lock()
//	-----------------------------------G1,G3 leak-------------------------------------

func Kubernetes10182() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 1000; i++ {
		go func() {
			s := NewStatusManager_kubernetes10182()
			go s.Start()
			go s.SetPodStatus()
			go s.SetPodStatus()
		}()
	}
}
