// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: moby
 * Issue or PR  : https://github.com/moby/moby/pull/28462
 * Buggy version: b184bdabf7a01c4b802304ac64ac133743c484be
 * fix commit-id: 89b123473774248fc3a0356dd3ce5b116cc69b29
 * Flaky: 69/100
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
	register("Moby28462", Moby28462)
}

type State_moby28462 struct {
	Health *Health_moby28462
}

type Container_moby28462 struct {
	sync.Mutex
	State *State_moby28462
}

func (ctr *Container_moby28462) start() {
	go ctr.waitExit()
}
func (ctr *Container_moby28462) waitExit() {

}

type Store_moby28462 struct {
	ctr *Container_moby28462
}

func (s *Store_moby28462) Get() *Container_moby28462 {
	return s.ctr
}

type Daemon_moby28462 struct {
	containers Store_moby28462
}

func (d *Daemon_moby28462) StateChanged() {
	c := d.containers.Get()
	c.Lock()
	d.updateHealthMonitorElseBranch(c)
	defer c.Unlock()
}

func (d *Daemon_moby28462) updateHealthMonitorIfBranch(c *Container_moby28462) {
	h := c.State.Health
	if stop := h.OpenMonitorChannel(); stop != nil {
		go monitor_moby28462(c, stop)
	}
}
func (d *Daemon_moby28462) updateHealthMonitorElseBranch(c *Container_moby28462) {
	h := c.State.Health
	h.CloseMonitorChannel()
}

type Health_moby28462 struct {
	stop chan struct{}
}

func (s *Health_moby28462) OpenMonitorChannel() chan struct{} {
	return s.stop
}

func (s *Health_moby28462) CloseMonitorChannel() {
	if s.stop != nil {
		s.stop <- struct{}{}
	}
}

func monitor_moby28462(c *Container_moby28462, stop chan struct{}) {
	for {
		select {
		case <-stop:
			return
		default:
			handleProbeResult_moby28462(c)
		}
	}
}

func handleProbeResult_moby28462(c *Container_moby28462) {
	runtime.Gosched()
	c.Lock()
	defer c.Unlock()
}

func NewDaemonAndContainer_moby28462() (*Daemon_moby28462, *Container_moby28462) {
	c := &Container_moby28462{
		State: &State_moby28462{&Health_moby28462{make(chan struct{})}},
	}
	d := &Daemon_moby28462{Store_moby28462{c}}
	return d, c
}

func Moby28462() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() {
			d, c := NewDaemonAndContainer_moby28462()
			go monitor_moby28462(c, c.State.Health.OpenMonitorChannel()) // G1
			go d.StateChanged()                                          // G2
		}()
	}
}
