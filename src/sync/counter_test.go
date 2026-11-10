// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"math"
	"runtime"
	"sync"
	"testing"
)

func equal[T ~int64 | ~uint64](a, b T) bool {
	return a == b
}

func testCounterAdd[T ~int64 | ~uint64 | ~float64](t *testing.T, checkVal func(got, want T) bool) {
	wantCount := int64(100000)
	if testing.Short() {
		wantCount = 1000
	}

	runAdders := func(want int64, gCount int) {
		var c sync.Counter[T]
		var wg sync.WaitGroup
		for range gCount {
			wg.Go(func() {
				for range want {
					c.Add(1)
				}
			})
		}
		wg.Wait()
		wantTotal := want * int64(gCount)
		if got := c.Value(); !checkVal(got, T(wantTotal)) {
			t.Errorf("Add want %v, got %v", wantTotal, got)
		}
	}

	t.Run("single goroutine add", func(t *testing.T) {
		runAdders(wantCount, 1)
	})

	t.Run("GOMAXPROCS goroutines adding", func(t *testing.T) {
		runAdders(wantCount, runtime.GOMAXPROCS(0))
	})

	t.Run("GOMAXPROCS multiple goroutines adding", func(t *testing.T) {
		runAdders(wantCount, runtime.GOMAXPROCS(0)*4)
	})
}

func testCountersAdd[T ~int64 | ~uint64 | ~float64](t *testing.T, checkVal func(got, want T) bool) {
	wantCount := int64(10000)
	outer := 100
	if testing.Short() {
		wantCount = 1000
		outer = 10
	}

	runAdders := func(outerWG *sync.WaitGroup, want int64, gCount int) {
		defer outerWG.Done()

		var c sync.Counter[T]
		var wg sync.WaitGroup
		for range gCount {
			wg.Go(func() {
				for range want {
					c.Add(1)
				}
			})
		}
		wg.Wait()
		wantTotal := want * int64(gCount)
		if got := c.Value(); !checkVal(got, T(wantTotal)) {
			t.Errorf("Add want %v, got %v", wantTotal, got)
		}
	}

	t.Run("single goroutine add", func(t *testing.T) {
		var wg sync.WaitGroup
		for range outer {
			wg.Add(1)
			go runAdders(&wg, wantCount, 1)
		}
		wg.Wait()
	})

	t.Run("GOMAXPROCS goroutines adding", func(t *testing.T) {
		var wg sync.WaitGroup
		for range outer {
			wg.Add(1)
			go runAdders(&wg, wantCount, runtime.GOMAXPROCS(0))
		}
		wg.Wait()
	})

	t.Run("GOMAXPROCS multiple goroutines adding", func(t *testing.T) {
		var wg sync.WaitGroup
		for range outer {
			wg.Add(1)
			go runAdders(&wg, wantCount, runtime.GOMAXPROCS(0)*4)
		}
		wg.Wait()
	})
}

func TestCounterAddInt64(t *testing.T) {
	testCounterAdd(t, equal[int64])
}

func TestCountersAddInt64(t *testing.T) {
	testCountersAdd(t, equal[int64])
}

func TestCounterAddUint64(t *testing.T) {
	testCounterAdd(t, equal[uint64])
}

func TestCountersAddUint64(t *testing.T) {
	testCountersAdd(t, equal[uint64])
}

func TestCounterUint64WrapAround(t *testing.T) {
	var c sync.Counter[uint64]
	c.Add(math.MaxUint64)
	c.Add(5)
	if got, want := c.Value(), uint64(4); got != want {
		t.Errorf("Value after wrap around: want %d, got %d", want, got)
	}
}

func TestCounterUint64WrapAroundParallel(t *testing.T) {
	var c sync.Counter[uint64]
	c.Add(math.MaxUint64 - 100)

	var wg sync.WaitGroup
	for range 105 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c.Add(1)
		}()
	}
	wg.Wait()

	if got, want := c.Value(), uint64(4); got != want {
		t.Errorf("Value after parallel wrap around: want %d, got %d", want, got)
	}
}

func TestCounterAddFloat64(t *testing.T) {
	testCounterAdd(t, floatEqual)
}

func TestCountersAddFloat64(t *testing.T) {
	testCountersAdd(t, floatEqual)
}

func TestCounterZeroValue(t *testing.T) {
	var ci sync.Counter[int64]
	if got := ci.Value(); got != 0 {
		t.Errorf("zero Counter[int64].Value() = %d, want 0", got)
	}
	var cu sync.Counter[uint64]
	if got := cu.Value(); got != 0 {
		t.Errorf("zero Counter[uint64].Value() = %d, want 0", got)
	}
	var cf sync.Counter[float64]
	if got := cf.Value(); got != 0 {
		t.Errorf("zero Counter[float64].Value() = %v, want 0", got)
	}
}

func TestCounterNamedTypes(t *testing.T) {
	type myInt64 int64
	type myUint64 uint64
	type myFloat64 float64

	testCounterAdd(t, equal[myInt64])
	testCounterAdd(t, equal[myUint64])
	testCounterAdd(t, func(got, want myFloat64) bool {
		return floatEqual(float64(got), float64(want))
	})
}

func TestCounterInt64NegativeAndWrapAround(t *testing.T) {
	t.Run("negative deltas", func(t *testing.T) {
		var c sync.Counter[int64]
		cpus := int64(runtime.GOMAXPROCS(0))
		const iters = 1000
		var wg sync.WaitGroup
		for range cpus {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for range iters {
					c.Add(2)
					c.Add(-5)
				}
			}()
		}
		wg.Wait()
		want := cpus * iters * (-3)
		if got := c.Value(); got != want {
			t.Errorf("Value() = %d, want %d", got, want)
		}
	})

	t.Run("wrap around MaxInt64 and MinInt64", func(t *testing.T) {
		var c sync.Counter[int64]
		c.Add(math.MaxInt64)
		c.Add(1)
		if got, want := c.Value(), int64(math.MinInt64); got != want {
			t.Errorf("after MaxInt64 + 1: got %d, want %d", got, want)
		}
		c.Add(-1)
		if got, want := c.Value(), int64(math.MaxInt64); got != want {
			t.Errorf("after MinInt64 - 1: got %d, want %d", got, want)
		}
	})
}

func TestCounterFloat64FractionalAndNegative(t *testing.T) {
	var c sync.Counter[float64]
	cpus := runtime.GOMAXPROCS(0)
	const iters = 1000
	var wg sync.WaitGroup
	for range cpus {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range iters {
				c.Add(0.1)
				c.Add(-0.25)
			}
		}()
	}
	wg.Wait()

	want := float64(cpus*iters) * (0.1 - 0.25)
	if got := c.Value(); !floatEqual(got, want) {
		t.Errorf("Value() = %v, want %v", got, want)
	}
}

func floatEqual(a, b float64) bool {
	if a == b {
		return true
	}
	const epsilon = 1e-9
	diff := math.Abs(a - b)
	if a == 0 || b == 0 || diff < math.SmallestNonzeroFloat64 {
		return diff <= epsilon
	}
	return diff <= epsilon*math.Max(math.Abs(a), math.Abs(b))
}

// TestCounterValueConcurrentWithAdd checks that an unsynchronized Value
// observes a total between the last synchronized value and the final value
// while Adds are in progress, and that successive Values from the same
// goroutine never decrease when all deltas are positive.
func TestCounterValueConcurrentWithAdd(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("skipping on wasm: the spinning reader is never preempted")
	}
	const phase = 1000
	procs := runtime.GOMAXPROCS(0)
	var c sync.Counter[int64]

	// addConcurrently spreads total Add(1) calls over GOMAXPROCS goroutines.
	addConcurrently := func(wg *sync.WaitGroup, total int) {
		for g := range procs {
			n := total / procs
			if g < total%procs {
				n++
			}
			wg.Go(func() {
				for range n {
					c.Add(1)
				}
			})
		}
	}

	// Phase 1: synchronize via WaitGroup, not via Value.
	var wg sync.WaitGroup
	addConcurrently(&wg, phase)
	wg.Wait()

	// Phase 2: read with Value while more Adds are in flight.
	var adders sync.WaitGroup
	addConcurrently(&adders, phase)
	done := make(chan struct{})
	var reader sync.WaitGroup
	reader.Go(func() {
		var prev int64
		for {
			got := c.Value()
			if got < phase || got > 2*phase {
				t.Errorf("concurrent Value() = %d, want in [%d, %d]", got, phase, 2*phase)
				return
			}
			if got < prev {
				t.Errorf("concurrent Value() decreased: %d after %d", got, prev)
				return
			}
			prev = got
			select {
			case <-done:
				return
			default:
			}
		}
	})
	adders.Wait()
	close(done)
	reader.Wait()

	if got := c.Value(); got != 2*phase {
		t.Errorf("final Value() = %d, want %d", got, 2*phase)
	}
}
