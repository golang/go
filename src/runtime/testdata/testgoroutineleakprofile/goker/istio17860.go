// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"os"
	"runtime/pprof"

	"sync"
	"time"
)

func init() {
	register("Istio17860", Istio17860)
}

type Proxy_istio17860 interface {
	IsLive() bool
}

type TestProxy_istio17860 struct {
	live func() bool
}

func (tp TestProxy_istio17860) IsLive() bool {
	if tp.live == nil {
		return true
	}
	return tp.live()
}

type Agent_istio17860 interface {
	Run(ctx context.Context)
	Restart()
}

type exitStatus_istio17860 int

type agent_istio17860 struct {
	proxy        Proxy_istio17860
	mu           *sync.Mutex
	statusCh     chan exitStatus_istio17860
	currentEpoch int
	activeEpochs map[int]struct{}
}

func (a *agent_istio17860) Run(ctx context.Context) {
	for {
		select {
		case status := <-a.statusCh:
			a.mu.Lock()
			delete(a.activeEpochs, int(status))
			active := len(a.activeEpochs)
			a.mu.Unlock()
			if active == 0 {
				return
			}
		case <-ctx.Done():
			return
		}
	}
}

func (a *agent_istio17860) Restart() {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.waitUntilLive()
	a.currentEpoch++
	a.activeEpochs[a.currentEpoch] = struct{}{}

	go a.runWait(a.currentEpoch)
}

func (a *agent_istio17860) runWait(epoch int) {
	a.statusCh <- exitStatus_istio17860(epoch)
}

func (a *agent_istio17860) waitUntilLive() {
	if len(a.activeEpochs) == 0 {
		return
	}

	interval := time.NewTicker(30 * time.Nanosecond)
	timer := time.NewTimer(100 * time.Nanosecond)
	defer func() {
		interval.Stop()
		timer.Stop()
	}()

	if a.proxy.IsLive() {
		return
	}

	for {
		select {
		case <-timer.C:
			return
		case <-interval.C:
			if a.proxy.IsLive() {
				return
			}
		}
	}
}

func NewAgent_istio17860(proxy Proxy_istio17860) Agent_istio17860 {
	return &agent_istio17860{
		proxy:        proxy,
		mu:           &sync.Mutex{},
		statusCh:     make(chan exitStatus_istio17860),
		activeEpochs: make(map[int]struct{}),
	}
}

func Istio17860() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			neverLive := func() bool {
				return false
			}

			a := NewAgent_istio17860(TestProxy_istio17860{live: neverLive})
			go func() { a.Run(ctx) }()

			a.Restart()
			go a.Restart()

			time.Sleep(200 * time.Nanosecond)
		}()
	}
}
