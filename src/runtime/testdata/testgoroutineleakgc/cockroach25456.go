package main

import (
	"os"
	"runtime/pprof"
	"time"
)

func init() {
	register("Cockroach25456", Cockroach25456)
}

type Stopper_cockroach25456 struct {
	quiescer chan struct{}
}

func (s *Stopper_cockroach25456) ShouldQuiesce() <-chan struct{} {
	if s == nil {
		return nil
	}
	return s.quiescer
}

func NewStopper_cockroach25456() *Stopper_cockroach25456 {
	return &Stopper_cockroach25456{quiescer: make(chan struct{})}
}

type Store_cockroach25456 struct {
	stopper          *Stopper_cockroach25456
	consistencyQueue *consistencyQueue_cockroach25456
}

func (s *Store_cockroach25456) Stopper() *Stopper_cockroach25456 {
	return s.stopper
}
func (s *Store_cockroach25456) Start(stopper *Stopper_cockroach25456) {
	s.stopper = stopper
}

func NewStore_cockroach25456() *Store_cockroach25456 {
	return &Store_cockroach25456{
		consistencyQueue: newConsistencyQueue_cockroach25456(),
	}
}

type Replica_cockroach25456 struct {
	store *Store_cockroach25456
}

func NewReplica_cockroach25456(store *Store_cockroach25456) *Replica_cockroach25456 {
	return &Replica_cockroach25456{store: store}
}

type consistencyQueue_cockroach25456 struct{}

func (q *consistencyQueue_cockroach25456) process(repl *Replica_cockroach25456) {
	<-repl.store.Stopper().ShouldQuiesce()
}

func newConsistencyQueue_cockroach25456() *consistencyQueue_cockroach25456 {
	return &consistencyQueue_cockroach25456{}
}

type testContext_cockroach25456 struct {
	store *Store_cockroach25456
	repl  *Replica_cockroach25456
}

func (tc *testContext_cockroach25456) StartWithStoreConfig(stopper *Stopper_cockroach25456) {
	if tc.store == nil {
		tc.store = NewStore_cockroach25456()
	}
	tc.store.Start(stopper)
	tc.repl = NewReplica_cockroach25456(tc.store)
}

func Cockroach25456() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	go func() {
		// deadlocks: 1
		stopper := NewStopper_cockroach25456()
		tc := testContext_cockroach25456{}
		tc.StartWithStoreConfig(stopper)

		for i := 0; i < 2; i++ {
			tc.store.consistencyQueue.process(tc.repl)
		}
	}()
}
