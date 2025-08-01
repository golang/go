/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/10790
 * Buggy version: 96b5452557ebe26bd9d85fe7905155009204d893
 * fix commit-id: f1a5c19125c65129b966fbdc0e6408e8df214aba
 * Flaky: 28/100
 * Description:
 *   It is possible that a message from ctxDone will make the function beginCmds
 * returns without draining the channel ch, so that goroutines created by anonymous
 * function will leak.
 */

package main

import (
	"context"
	"runtime"
	"sync"
	"time"
)

func init() {
	register("Cockroach10790", Cockroach10790)
}

type Stopper_cockroach10790 struct {
	quiescer chan struct{}
	mu       struct {
		sync.Mutex
		quiescing bool
	}
}

func (s *Stopper_cockroach10790) ShouldQuiesce() <-chan struct{} {
	if s == nil {
		return nil
	}
	return s.quiescer
}

func (s *Stopper_cockroach10790) Quiesce() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.mu.quiescing {
		s.mu.quiescing = true
		close(s.quiescer)
	}
}

func (s *Stopper_cockroach10790) Stop() {
	s.Quiesce()
}

type Replica_cockroach10790 struct {
	chans   []chan bool
	stopper *Stopper_cockroach10790
}

func (r *Replica_cockroach10790) beginCmds(ctx context.Context) {
	ctxDone := ctx.Done()
	for _, ch := range r.chans {
		select {
		case <-ch:
		case <-ctxDone:
			go func() {
				// deadlocks: x > 0
				for _, ch := range r.chans {
					<-ch
				}
			}()
		}
	}
}

func (r *Replica_cockroach10790) sendChans(ctx context.Context) {
	for _, ch := range r.chans {
		select {
		case ch <- true:
		case <-ctx.Done():
			return
		}
	}
}

func NewReplica_cockroach10790() *Replica_cockroach10790 {
	r := &Replica_cockroach10790{
		stopper: &Stopper_cockroach10790{
			quiescer: make(chan struct{}),
		},
	}
	r.chans = append(r.chans, make(chan bool))
	r.chans = append(r.chans, make(chan bool))
	return r
}

///
/// G1					G2				helper goroutine
/// 									r.sendChans()
/// r.beginCmds()
/// 									ch1 <- true
/// <- ch1
///										ch2 <- true
///	...					...				...
///						cancel()
///	<- ch1
///	------------------G1 leak--------------------------
///

func Cockroach10790() {
	defer func() {
		time.Sleep(100 * time.Millisecond)
		runtime.GC()
	}()

	for i := 0; i < 100; i++ {
		go func() {
			r := NewReplica_cockroach10790()
			ctx, cancel := context.WithCancel(context.Background())
			go r.sendChans(ctx) // helper goroutine
			go r.beginCmds(ctx) // G1
			go cancel()         // G2
			r.stopper.Stop()
		}()
	}
}
