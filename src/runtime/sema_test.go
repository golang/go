// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	. "runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// TestSemaHandoff checks that when semrelease+handoff is
// requested, the G that releases the semaphore yields its
// P directly to the first waiter in line.
// See issue 33747 for discussion.
func TestSemaHandoff(t *testing.T) {
	const iter = 10000
	ok := 0
	for i := 0; i < iter; i++ {
		if testSemaHandoff() {
			ok++
		}
	}
	// As long as two thirds of handoffs are direct, we
	// consider the test successful. The scheduler is
	// nondeterministic, so this test checks that we get the
	// desired outcome in a significant majority of cases.
	// The actual ratio of direct handoffs is much higher
	// (>90%) but we use a lower threshold to minimize the
	// chances that unrelated changes in the runtime will
	// cause the test to fail or become flaky.
	if ok < iter*2/3 {
		t.Fatal("direct handoff < 2/3:", ok, iter)
	}
}

func TestSemaHandoff1(t *testing.T) {
	if GOMAXPROCS(-1) <= 1 {
		t.Skip("GOMAXPROCS <= 1")
	}
	defer GOMAXPROCS(GOMAXPROCS(-1))
	GOMAXPROCS(1)
	TestSemaHandoff(t)
}

func TestSemaHandoff2(t *testing.T) {
	if GOMAXPROCS(-1) <= 2 {
		t.Skip("GOMAXPROCS <= 2")
	}
	defer GOMAXPROCS(GOMAXPROCS(-1))
	GOMAXPROCS(2)
	TestSemaHandoff(t)
}

func testSemaHandoff() bool {
	var sema, res uint32
	done := make(chan struct{})

	// We're testing that the current goroutine is able to yield its time slice
	// to another goroutine. Stop the current goroutine from migrating to
	// another CPU where it can win the race (and appear to have not yielded) by
	// keeping the CPUs slightly busy.
	var wg sync.WaitGroup
	for i := 0; i < GOMAXPROCS(-1); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-done:
					return
				default:
				}
				Gosched()
			}
		}()
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		Semacquire(&sema)
		atomic.CompareAndSwapUint32(&res, 0, 1)

		Semrelease1(&sema, true, 0)
		close(done)
	}()
	for SemNwait(&sema) == 0 {
		Gosched() // wait for goroutine to block in Semacquire
	}

	// The crux of the test: we release the semaphore with handoff
	// and immediately perform a CAS both here and in the waiter; we
	// want the CAS in the waiter to execute first.
	Semrelease1(&sema, true, 0)
	atomic.CompareAndSwapUint32(&res, 0, 2)

	wg.Wait() // wait for goroutines to finish to avoid data races

	return res == 1 // did the waiter run first?
}

func BenchmarkSemTable(b *testing.B) {
	for _, n := range []int{1000, 2000, 4000, 8000} {
		b.Run(fmt.Sprintf("OneAddrCollision/n=%d", n), func(b *testing.B) {
			tab := Escape(new(SemTable))
			u := make([]uint32, SemTableSize+1)

			b.ResetTimer()

			for j := 0; j < b.N; j++ {
				// Simulate two locks colliding on the same semaRoot.
				//
				// Specifically enqueue all the waiters for the first lock,
				// then all the waiters for the second lock.
				//
				// Then, dequeue all the waiters from the first lock, then
				// the second.
				//
				// Each enqueue/dequeue operation should be O(1), because
				// there are exactly 2 locks. This could be O(n) if all
				// the waiters for both locks are on the same list, as it
				// once was.
				for i := 0; i < n; i++ {
					if i < n/2 {
						tab.Enqueue(&u[0])
					} else {
						tab.Enqueue(&u[SemTableSize])
					}
				}
				for i := 0; i < n; i++ {
					var ok bool
					if i < n/2 {
						ok = tab.Dequeue(&u[0])
					} else {
						ok = tab.Dequeue(&u[SemTableSize])
					}
					if !ok {
						b.Fatal("failed to dequeue")
					}
				}
			}
		})
		b.Run(fmt.Sprintf("ManyAddrCollision/n=%d", n), func(b *testing.B) {
			tab := Escape(new(SemTable))
			u := make([]uint32, n*SemTableSize)

			b.ResetTimer()

			for j := 0; j < b.N; j++ {
				// Simulate n locks colliding on the same semaRoot.
				//
				// Each enqueue/dequeue operation should be O(log n), because
				// each semaRoot is a tree. This could be O(n) if it was
				// some simpler data structure.
				for i := 0; i < n; i++ {
					tab.Enqueue(&u[i*SemTableSize])
				}
				for i := 0; i < n; i++ {
					if !tab.Dequeue(&u[i*SemTableSize]) {
						b.Fatal("failed to dequeue")
					}
				}
			}
		})
	}
}
