package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
)

func init() {
	register("Cockroach584", Cockroach584)
}

type gossip_cockroach584 struct {
	mu     sync.Mutex
	closed bool
}

func (g *gossip_cockroach584) bootstrap() {
	for {
		g.mu.Lock()
		if g.closed {
			/// Missing g.mu.Unlock
			break
		}
		g.mu.Unlock()
	}
}

func (g *gossip_cockroach584) manage() {
	for {
		g.mu.Lock()
		if g.closed {
			/// Missing g.mu.Unlock
			break
		}
		g.mu.Unlock()
	}
}

func Cockroach584() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		for i := 0; i < yieldCount; i++ {
			// Yield several times to allow the child goroutine to run.
			runtime.Gosched()
		}
		prof.WriteTo(os.Stdout, 2)
	}()

	g := &gossip_cockroach584{
		closed: true,
	}
	go func() {
		// deadlocks: 1
		g.bootstrap()
		g.manage()
	}()
}
