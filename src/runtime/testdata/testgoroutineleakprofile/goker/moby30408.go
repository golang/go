// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
)

func init() {
	register("Moby30408", Moby30408)
}

type Manifest_moby30408 struct {
	Implements []string
}

type Plugin_moby30408 struct {
	activateWait *sync.Cond
	activateErr  error
	Manifest     *Manifest_moby30408
}

func (p *Plugin_moby30408) waitActive() error {
	p.activateWait.L.Lock()
	for !p.activated() {
		p.activateWait.Wait()
	}
	p.activateWait.L.Unlock()
	return p.activateErr
}

func (p *Plugin_moby30408) activated() bool {
	return p.Manifest != nil
}

func testActive_moby30408(p *Plugin_moby30408) {
	done := make(chan struct{})
	go func() { // G2
		p.waitActive()
		close(done)
	}()
	<-done
}

func Moby30408() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()

	go func() { // G1
		p := &Plugin_moby30408{activateWait: sync.NewCond(&sync.Mutex{})}
		p.activateErr = errors.New("some junk happened")

		testActive_moby30408(p)
	}()
}
