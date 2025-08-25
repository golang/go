package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
)

func init() {
	register("Moby29733", Moby29733)
}

type Plugin_moby29733 struct {
	activated    bool
	activateWait *sync.Cond
}

type plugins_moby29733 struct {
	sync.Mutex
	plugins map[int]*Plugin_moby29733
}

func (p *Plugin_moby29733) waitActive() {
	p.activateWait.L.Lock()
	for !p.activated {
		p.activateWait.Wait()
	}
	p.activateWait.L.Unlock()
}

type extpointHandlers_moby29733 struct {
	sync.RWMutex
	extpointHandlers map[int]struct{}
}

func Handle_moby29733(storage plugins_moby29733, handlers extpointHandlers_moby29733) {
	handlers.Lock()
	for _, p := range storage.plugins {
		p.activated = false
	}
	handlers.Unlock()
}

func testActive_moby29733(p *Plugin_moby29733) {
	done := make(chan struct{})
	go func() {
		// deadlocks: x > 0
		p.waitActive()
		close(done)
	}()
	<-done
}

func Moby29733() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		// Yield several times to allow the child goroutine to run.
		for i := 0; i < yieldCount; i++ {
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()

	go func() {
		// deadlocks: x > 0
		storage := plugins_moby29733{plugins: make(map[int]*Plugin_moby29733)}
		handlers := extpointHandlers_moby29733{extpointHandlers: make(map[int]struct{})}

		p := &Plugin_moby29733{activateWait: sync.NewCond(&sync.Mutex{})}
		storage.plugins[0] = p

		testActive_moby29733(p)
		Handle_moby29733(storage, handlers)
		testActive_moby29733(p)
	}()
}
