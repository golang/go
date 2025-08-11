/*
 * Project: moby
 * Issue or PR  : https://github.com/moby/moby/pull/25384
 * Buggy version: 58befe3081726ef74ea09198cd9488fb42c51f51
 * fix commit-id: 42360d164b9f25fb4b150ef066fcf57fa39559a7
 * Flaky: 100/100
 * Description:
 *   When n=1 (len(pm.plugins)), the location of group.Wait() doesn’t matter.
 * When n is larger than 1, group.Wait() is invoked in each iteration. Whenever
 * group.Wait() is invoked, it waits for group.Done() to be executed n times.
 * However, group.Done() is only executed once in one iteration.
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
		runtime.Gosched()
		prof.WriteTo(os.Stdout, 2)
	}()
	go func() {
		p1 := &plugin_moby25348{}
		p2 := &plugin_moby25348{}
		pm := &Manager_moby25348{
			plugins: []*plugin_moby25348{p1, p2},
		}
		// deadlocks: 1
		go pm.init()
	}()
}
