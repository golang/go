// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fuzz provides common fuzzing functionality for tests built with
// "go test" and for programs that use fuzzing functionality in the testing
// package.
package fuzz

import (
	"os"
	"runtime"
	"sync"
	"time"
)

// CoordinateFuzzing creates several worker processes and communicates with
// them to test random inputs that could trigger crashes and expose bugs.
// The worker processes run the same binary in the same directory with the
// same environment variables as the coordinator process. Workers also run
// with the same arguments as the coordinator, except with the -test.fuzzworker
// flag prepended to the argument list.
//
// parallel is the number of worker processes to run in parallel. If parallel
// is 0, CoordinateFuzzing will run GOMAXPROCS workers.
//
// seed is a list of seed values added by the fuzz target with testing.F.Add.
// Seed values from testdata and GOFUZZCACHE should not be included in this
// list; this function loads them separately.
func CoordinateFuzzing(parallel int, seed [][]byte) error {
	if parallel == 0 {
		parallel = runtime.GOMAXPROCS(0)
	}
	// TODO(jayconrod): support fuzzing indefinitely or with a given duration.
	// The value below is just a placeholder until we figure out how to handle
	// interrupts.
	duration := 5 * time.Second

	// TODO(jayconrod): do we want to support fuzzing different binaries?
	dir := "" // same as self
	binPath := os.Args[0]
	args := append([]string{"-test.fuzzworker"}, os.Args[1:]...)
	env := os.Environ() // same as self

	c := &coordinator{
		doneC:  make(chan struct{}),
		inputC: make(chan corpusEntry),
	}

	newWorker := func() *worker {
		return &worker{
			dir:         dir,
			binPath:     binPath,
			args:        args,
			env:         env,
			coordinator: c,
		}
	}

	corpus := corpus{entries: make([]corpusEntry, len(seed))}
	for i, v := range seed {
		corpus.entries[i].b = v
	}
	if len(corpus.entries) == 0 {
		// TODO(jayconrod,katiehockman): pick a good starting corpus when one is
		// missing or very small.
		corpus.entries = append(corpus.entries, corpusEntry{b: []byte{0}})
	}

	// TODO(jayconrod,katiehockman): read corpus from testdata.
	// TODO(jayconrod,katiehockman): read corpus from GOFUZZCACHE.

	// Start workers.
	workers := make([]*worker, parallel)
	runErrs := make([]error, parallel)
	var wg sync.WaitGroup
	wg.Add(parallel)
	for i := 0; i < parallel; i++ {
		go func(i int) {
			defer wg.Done()
			workers[i] = newWorker()
			runErrs[i] = workers[i].runFuzzing()
		}(i)
	}

	// Main event loop.
	stopC := time.After(duration)
	i := 0
	for {
		select {
		// TODO(jayconrod): handle interruptions like SIGINT.
		// TODO(jayconrod,katiehockman): receive crashers and new corpus values
		// from workers.

		case <-stopC:
			// Time's up.
			close(c.doneC)

		case <-c.doneC:
			// Wait for workers to stop and return.
			wg.Wait()
			for _, err := range runErrs {
				if err != nil {
					return err
				}
			}
			return nil

		case c.inputC <- corpus.entries[i]:
			// Sent the next input to any worker.
			// TODO(jayconrod,katiehockman): need a scheduling algorithm that chooses
			// which corpus value to send next (or generates something new).
			i = (i + 1) % len(corpus.entries)
		}
	}

	// TODO(jayconrod,katiehockman): write crashers to testdata and other inputs
	// to GOFUZZCACHE. If the testdata directory is outside the current module,
	// always write to GOFUZZCACHE, since the testdata is likely read-only.
}

type corpus struct {
	entries []corpusEntry
}

// TODO(jayconrod,katiehockman): decide whether and how to unify this type
// with the equivalent in testing.
type corpusEntry struct {
	b []byte
}

// coordinator holds channels that workers can use to communicate with
// the coordinator.
type coordinator struct {
	// doneC is closed to indicate fuzzing is done and workers should stop.
	// doneC may be closed due to a time limit expiring or a fatal error in
	// a worker.
	doneC chan struct{}

	// inputC is sent values to fuzz by the coordinator. Any worker may receive
	// values from this channel.
	inputC chan corpusEntry
}
