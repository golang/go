package main

import (
	"os"
	"runtime/pprof"
	"sync"
	"time"
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
		time.Sleep(10 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() {
			g := &gossip_cockroach584{
				closed: true,
			}
			go func() {
				// deadlocks: x > 0
				g.bootstrap()
				g.manage()
			}()
		}()
	}
}
