// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: moby
 * Issue or PR  : https://github.com/moby/moby/pull/25384
 * Buggy version: 58befe3081726ef74ea09198cd9488fb42c51f51
 * fix commit-id: 42360d164b9f25fb4b150ef066fcf57fa39559a7
 * Flaky: 100/100
 */
package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
)

func init() {
	register("Moby25348", Moby25348)
}

type plugin_moby25348 struct{}

type Manager_moby25348 struct {
	plugins []*plugin_moby25348
}

func (pm *Manager_moby25348) init() {
	var group sync.WaitGroup
	group.Add(len(pm.plugins))
	for _, p := range pm.plugins {
		go func(p *plugin_moby25348) {
			defer group.Done()
		}(p)
		group.Wait() // Block here
	}
}

func Moby25348() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()
	go func() {
		p1 := &plugin_moby25348{}
		p2 := &plugin_moby25348{}
		pm := &Manager_moby25348{
			plugins: []*plugin_moby25348{p1, p2},
		}
		go pm.init()
	}()
}
