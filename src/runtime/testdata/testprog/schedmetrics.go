// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"log"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/metrics"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

func init() {
	register("SchedMetrics", SchedMetrics)
}

// Tests runtime/metrics.Read for various scheduler metrics.
//
// Implemented in testprog to prevent other tests from polluting
// the metrics.
func SchedMetrics() {
	const (
		notInGo = iota
		runnable
		running
		waiting
		created
		threads
		numSamples
	)
	var s [numSamples]metrics.Sample
	s[notInGo].Name = "/sched/goroutines/not-in-go:goroutines"
	s[runnable].Name = "/sched/goroutines/runnable:goroutines"
	s[running].Name = "/sched/goroutines/running:goroutines"
	s[waiting].Name = "/sched/goroutines/waiting:goroutines"
	s[created].Name = "/sched/goroutines-created:goroutines"
	s[threads].Name = "/sched/threads/total:threads"

	var failed bool
	var out bytes.Buffer
	logger := log.New(&out, "", 0)
	indent := 0
	logf := func(s string, a ...any) {
		var prefix strings.Builder
		for range indent {
			prefix.WriteString("\t")
		}
		logger.Printf(prefix.String()+s, a...)
	}
	errorf := func(s string, a ...any) {
		logf(s, a...)
		failed = true
	}
	run := func(name string, f func()) {
		logf("=== Checking %q", name)
		indent++
		f()
		indent--
	}
	logMetrics := func(s []metrics.Sample) {
		for i := range s {
			logf("%s: %d", s[i].Name, s[i].Value.Uint64())
		}
	}

	initialGMP := runtime.GOMAXPROCS(-1)
	logf("Initial GOMAXPROCS=%d", initialGMP)

	// generalSlack is the amount of goroutines we allow ourselves to be
	// off by in any given category, either due to background system
	// goroutines. This excludes GC goroutines.
	generalSlack := uint64(4)

	// waitingSlack is the max number of blocked goroutines controlled
	// by the runtime that we'll allow for. This includes GC goroutines
	// as well as finalizer and cleanup goroutines.
	waitingSlack := generalSlack + uint64(2*initialGMP)

	// threadsSlack is the maximum number of threads left over
	// from the runtime (sysmon, the template thread, etc.)
	// Certain build modes may also cause the creation of additional
	// threads through frequent scheduling, like mayMoreStackPreempt.
	// A slack of 5 is arbitrary but appears to be enough to cover
	// the leftovers plus any inflation from scheduling-heavy build
	// modes.
	const threadsSlack = 5

	// Make sure GC isn't running, since GC workers interfere with
	// expected counts.
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	runtime.GC()

	check := func(s *metrics.Sample, min, max uint64) {
		val := s.Value.Uint64()
		if val < min {
			errorf("%s too low; %d < %d", s.Name, val, min)
		}
		if val > max {
			errorf("%s too high; %d > %d", s.Name, val, max)
		}
	}
	checkEq := func(s *metrics.Sample, value uint64) {
		check(s, value, value)
	}
	spinUntil := func(f func() bool) bool {
		for {
			if f() {
				return true
			}
			time.Sleep(50 * time.Millisecond)
		}
	}

	// Check base values.
	run("base", func() {
		defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
		metrics.Read(s[:])
		logMetrics(s[:])
		check(&s[notInGo], 0, generalSlack)
		check(&s[runnable], 0, generalSlack)
		checkEq(&s[running], 1)
		check(&s[waiting], 0, waitingSlack)
	})

	metrics.Read(s[:])
	createdAfterBase := s[created].Value.Uint64()

	// Force Running count to be high. We'll use these goroutines
	// for Runnable, too.
	const count = 10
	var ready, exit atomic.Uint32
	for range count {
		go func() {
			ready.Add(1)
			for exit.Load() == 0 {
				// Spin to get us and keep us running, but check
				// the exit condition so we exit out early if we're
				// done.
				start := time.Now()
				for time.Since(start) < 10*time.Millisecond && exit.Load() == 0 {
				}
				runtime.Gosched()
			}
		}()
	}
	for ready.Load() < count {
		runtime.Gosched()
	}

	// Be careful. We've entered a dangerous state for platforms
	// that do not return back to the underlying system unless all
	// goroutines are blocked, like js/wasm, since we have a bunch
	// of runnable goroutines all spinning. We cannot write anything
	// out.
	if testenv.HasParallelism() {
		run("created", func() {
			metrics.Read(s[:])
			logMetrics(s[:])
			checkEq(&s[created], createdAfterBase+count)
		})
		run("running", func() {
			defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(count + 4))
			// It can take a little bit for the scheduler to
			// distribute the goroutines to Ps, so retry until
			// we see the count we expect or the test times out.
			spinUntil(func() bool {
				metrics.Read(s[:])
				return s[running].Value.Uint64() >= count
			})
			logMetrics(s[:])
			check(&s[running], count, count+4)
			check(&s[threads], count, count+4+threadsSlack)
		})

		// Force runnable count to be high.
		run("runnable", func() {
			defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
			metrics.Read(s[:])
			logMetrics(s[:])
			checkEq(&s[running], 1)
			check(&s[runnable], count-1, count+generalSlack)
		})

		// Done with the running/runnable goroutines.
		exit.Store(1)
	} else {
		// Read metrics and then exit all the other goroutines,
		// so that system calls may proceed.
		metrics.Read(s[:])

		// Done with the running/runnable goroutines.
		exit.Store(1)

		// Now we can check our invariants.
		run("created", func() {
			// Look for count-1 goroutines because we read metrics
			// *before* run goroutine was created for this sub-test.
			checkEq(&s[created], createdAfterBase+count-1)
		})
		run("running", func() {
			logMetrics(s[:])
			checkEq(&s[running], 1)
			checkEq(&s[threads], 1)
		})
		run("runnable", func() {
			logMetrics(s[:])
			check(&s[runnable], count-1, count+generalSlack)
		})
	}

	// Force not-in-go count to be high. This is a little tricky since
	// we try really hard not to let things block in system calls.
	// We have to drop to the syscall package to do this reliably.
	run("not-in-go", func() {
		// Block a bunch of goroutines on an OS pipe.
		pr, pw, err := pipe()
		if err != nil {
			switch runtime.GOOS {
			case "js", "wasip1":
				logf("creating pipe: %v", err)
				return
			}
			panic(fmt.Sprintf("creating pipe: %v", err))
		}
		for i := 0; i < count; i++ {
			go syscall.Read(pr, make([]byte, 1))
		}

		// Let the goroutines block.
		spinUntil(func() bool {
			metrics.Read(s[:])
			return s[notInGo].Value.Uint64() >= count
		})
		logMetrics(s[:])
		check(&s[notInGo], count, count+generalSlack)

		syscall.Close(pw)
		syscall.Close(pr)
	})

	run("waiting", func() {
		// Force waiting count to be high.
		const waitingCount = 1000
		stop := make(chan bool)
		for i := 0; i < waitingCount; i++ {
			go func() { <-stop }()
		}

		// Let the goroutines block.
		spinUntil(func() bool {
			metrics.Read(s[:])
			return s[waiting].Value.Uint64() >= waitingCount
		})
		logMetrics(s[:])
		check(&s[waiting], waitingCount, waitingCount+waitingSlack)

		close(stop)
	})

	if failed {
		fmt.Fprintln(os.Stderr, out.String())
		os.Exit(1)
	} else {
		fmt.Fprintln(os.Stderr, "OK")
	}
}
