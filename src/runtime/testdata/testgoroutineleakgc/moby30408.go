package main

import (
	"errors"
	"os"
	"runtime/pprof"
	"sync"
	"time"
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
	go func() {
		// deadlocks: 100
		p.waitActive()
		close(done)
	}()
	<-done
}

func Moby30408() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() {
			// deadlocks: 100
			p := &Plugin_moby30408{activateWait: sync.NewCond(&sync.Mutex{})}
			p.activateErr = errors.New("some junk happened")

			testActive_moby30408(p)
		}()
	}
}
