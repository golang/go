package sync_test

import (
	"runtime"
	. "sync"
	"sync/atomic"
	"testing"
)

func TestZeroSharded(t *testing.T) {
	var s Sharded[int]
	defer func() {
		p := recover()
		if p == nil {
			t.Fatal("use zero value Sharded should panic")
		}
	}()
	s.Get()
}

func TestCreateZeroSharded(t *testing.T) {
	defer func() {
		p := recover()
		if p == nil {
			t.Fatal("create zero value Sharded should panic")
		}
	}()
	var _ Sharded[int] = NewSharded[int](0)
}

func TestSharedRace(t *testing.T) {
	maxprocs := uint(runtime.GOMAXPROCS(0))
	var s = NewSharded[atomic.Int64](maxprocs)
	var wg WaitGroup
	wg.Add(int(maxprocs))
	for range maxprocs {
		go func() {
			defer wg.Done()
			for range 10000 {
				s.Get().Add(1)
			}
		}()
	}
	wg.Wait()
}

func TestSharedOneP(t *testing.T) {
	var s = NewSharded[atomic.Int64](uint(runtime.GOMAXPROCS(0)))
	for range 10000 {
		s.Get().Add(1)
	}
	if load := s.Get().Load(); load != int64(10000) {
		t.Fatalf("expected %d, got %d", 1000, load)
	}
}

func TestSharedRange(t *testing.T) {
	maxprocs := uint(runtime.GOMAXPROCS(0))
	var s = NewSharded[atomic.Int64](maxprocs)
	var wg WaitGroup
	wg.Add(int(maxprocs))
	for range maxprocs {
		go func() {
			defer wg.Done()
			for range 10000 {
				s.Get().Add(1)
			}
		}()
	}
	wg.Wait()
	i := int64(0)
	s.Range(func(v *atomic.Int64) {
		i += v.Load()
	})
	if i != 10000*int64(maxprocs) {
		t.Fatalf("Sharded use after got %d want %d", i, 10000*int64(maxprocs))
	}
}

func BenchmarkSharedAtomicInt64(b *testing.B) {
	maxprocs := uint(runtime.GOMAXPROCS(0))
	for range b.N {
		var s = NewSharded[atomic.Int64](maxprocs)
		var wg WaitGroup
		wg.Add(int(maxprocs))
		for range maxprocs {
			go func() {
				defer wg.Done()
				for range 10000 {
					s.Get().Add(1)
				}
			}()
		}
		wg.Wait()
	}
}

func ExampleSharded() {
	maxprocs := uint(runtime.GOMAXPROCS(0))
	// create a Sharded to use as a counter
	var s = NewSharded[atomic.Int64](maxprocs)
	var wg WaitGroup
	wg.Add(int(maxprocs))
	for range maxprocs {
		go func() {
			defer wg.Done()
			for range 10000 {
				// increment the value of the counter
				s.Get().Add(1)
			}
		}()
	}
	wg.Wait()
	// when no further counters are added
	// get the counter value
	i := int64(0)
	s.Range(func(v *atomic.Int64) {
		i += v.Load()
	})
}
