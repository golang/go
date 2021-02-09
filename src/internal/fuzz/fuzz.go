// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fuzz provides common fuzzing functionality for tests built with
// "go test" and for programs that use fuzzing functionality in the testing
// package.
package fuzz

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"sync"
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
// seed is a list of seed values added by the fuzz target with testing.F.Add and
// in testdata.
//
// corpusDir is a directory where files containing values that crash the
// code being tested may be written.
//
// cacheDir is a directory containing additional "interesting" values.
// The fuzzer may derive new values from these, and may write new values here.
//
// If a crash occurs, the function will return an error containing information
// about the crash, which can be reported to the user.
func CoordinateFuzzing(ctx context.Context, parallel int, seed []CorpusEntry, corpusDir, cacheDir string) (err error) {
	if err := ctx.Err(); err != nil {
		return err
	}
	if parallel == 0 {
		parallel = runtime.GOMAXPROCS(0)
	}

	sharedMemSize := 100 << 20 // 100 MB
	corpus, err := readCorpusAndCache(seed, corpusDir, cacheDir)
	if err != nil {
		return err
	}
	if len(corpus.entries) == 0 {
		corpus.entries = []CorpusEntry{{Data: []byte{}}}
	}

	// TODO(jayconrod): do we want to support fuzzing different binaries?
	dir := "" // same as self
	binPath := os.Args[0]
	args := append([]string{"-test.fuzzworker"}, os.Args[1:]...)
	env := os.Environ() // same as self

	c := &coordinator{
		doneC:        make(chan struct{}),
		inputC:       make(chan CorpusEntry),
		interestingC: make(chan CorpusEntry),
		crasherC:     make(chan crasherEntry),
		errC:         make(chan error),
	}

	newWorker := func() (*worker, error) {
		mem, err := sharedMemTempFile(sharedMemSize)
		if err != nil {
			return nil, err
		}
		memMu := make(chan *sharedMem, 1)
		memMu <- mem
		return &worker{
			dir:         dir,
			binPath:     binPath,
			args:        args,
			env:         env,
			coordinator: c,
			memMu:       memMu,
		}, nil
	}

	// Start workers.
	workers := make([]*worker, parallel)
	for i := range workers {
		var err error
		workers[i], err = newWorker()
		if err != nil {
			return err
		}
	}

	workerErrs := make([]error, len(workers))
	var wg sync.WaitGroup
	wg.Add(len(workers))
	for i := range workers {
		go func(i int) {
			defer wg.Done()
			workerErrs[i] = workers[i].runFuzzing()
			if cleanErr := workers[i].cleanup(); workerErrs[i] == nil {
				workerErrs[i] = cleanErr
			}
		}(i)
	}

	// Before returning, signal workers to stop, wait for them to actually stop,
	// and gather any errors they encountered.
	defer func() {
		close(c.doneC)
		wg.Wait()
		if err == nil || err == ctx.Err() {
			for _, werr := range workerErrs {
				if werr != nil {
					// Return the first error found, replacing ctx.Err() if a more
					// interesting error is found.
					err = werr
					break
				}
			}
		}
	}()

	// Main event loop.
	i := 0
	for {
		select {
		case <-ctx.Done():
			// Interrupted, cancelled, or timed out.
			// TODO(jayconrod,katiehockman): On Windows, ^C only interrupts 'go test',
			// not the coordinator or worker processes. 'go test' will stop running
			// actions, but it won't interrupt its child processes. This makes it
			// difficult to stop fuzzing on Windows without a timeout.
			return ctx.Err()

		case crasher := <-c.crasherC:
			// A worker found a crasher. Write it to testdata and return it.
			fileName, err := writeToCorpus(crasher.Data, corpusDir)
			if err == nil {
				err = &crashError{
					name: filepath.Base(fileName),
					err:  errors.New(crasher.errMsg),
				}
			}
			// TODO(jayconrod,katiehockman): if -keepfuzzing, report the error to
			// the user and restart the crashed worker.
			return err

		case entry := <-c.interestingC:
			// Some interesting input arrived from a worker.
			// This is not a crasher, but something interesting that should
			// be added to the on disk corpus and prioritized for future
			// workers to fuzz.
			// TODO(jayconrod, katiehockman): Prioritize fuzzing these values which
			// expanded coverage.
			// TODO(jayconrod, katiehockman): Don't write a value that's already
			// in the corpus.
			corpus.entries = append(corpus.entries, entry)
			if cacheDir != "" {
				if _, err := writeToCorpus(entry.Data, cacheDir); err != nil {
					return err
				}
			}

		case err := <-c.errC:
			// A worker encountered a fatal error.
			return err

		case c.inputC <- corpus.entries[i]:
			// Send the next input to any worker.
			// TODO(jayconrod,katiehockman): need a scheduling algorithm that chooses
			// which corpus value to send next (or generates something new).
			i = (i + 1) % len(corpus.entries)
		}
	}

	// TODO(jayconrod,katiehockman): if a crasher can't be written to corpusDir,
	// write to cacheDir instead.
}

// crashError wraps a crasher written to the seed corpus. It saves the name
// of the file where the input causing the crasher was saved. The testing
// framework uses this to report a command to re-run that specific input.
type crashError struct {
	name string
	err  error
}

func (e *crashError) Error() string {
	return e.err.Error()
}

func (e *crashError) Unwrap() error {
	return e.err
}

func (e *crashError) CrashName() string {
	return e.name
}

type corpus struct {
	entries []CorpusEntry
}

// CorpusEntry represents an individual input for fuzzing.
//
// We must use an equivalent type in the testing and testing/internal/testdeps
// packages, but testing can't import this package directly, and we don't want
// to export this type from testing. Instead, we use the same struct type and
// use a type alias (not a defined type) for convenience.
type CorpusEntry = struct {
	// Name is the name of the corpus file, if the entry was loaded from the
	// seed corpus. It can be used with -run. For entries added with f.Add and
	// entries generated by the mutator, Name is empty.
	Name string

	// Data is the raw data loaded from a corpus file.
	Data []byte

	// TODO(jayconrod,katiehockman): support multiple values of different types
	// added with f.Add with a Values []interface{} field. We'll need marhsalling
	// and unmarshalling functions, and we'll need to figure out what to do
	// in the mutator.
}

type crasherEntry struct {
	CorpusEntry
	errMsg string
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
	inputC chan CorpusEntry

	// interestingC is sent interesting values by the worker, which is received
	// by the coordinator. Values are usually interesting because they
	// increase coverage.
	interestingC chan CorpusEntry

	// crasherC is sent values that crashed the code being fuzzed. These values
	// should be saved in the corpus, and we may want to stop fuzzing after
	// receiving one.
	crasherC chan crasherEntry

	// errC is sent internal errors encountered by workers. When the coordinator
	// receives an error, it closes doneC and returns.
	errC chan error
}

// readCorpusAndCache creates a combined corpus from seed values, values in the
// corpus directory (in testdata), and values in the cache (in GOCACHE/fuzz).
//
// TODO(jayconrod,katiehockman): if a value in the cache has the wrong type,
// ignore it instead of reporting an error. Cached values may be used for
// the same package at a different version or in a different module.
// TODO(jayconrod,katiehockman): need a mechanism that can remove values that
// aren't useful anymore, for example, because they have the wrong type.
func readCorpusAndCache(seed []CorpusEntry, corpusDir, cacheDir string) (corpus, error) {
	var c corpus
	c.entries = append(c.entries, seed...)
	for _, dir := range []string{corpusDir, cacheDir} {
		entries, err := ReadCorpus(dir)
		if err != nil {
			return corpus{}, err
		}
		c.entries = append(c.entries, entries...)
	}
	return c, nil
}

// ReadCorpus reads the corpus from the testdata directory in this target's
// package.
func ReadCorpus(dir string) ([]CorpusEntry, error) {
	files, err := ioutil.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil // No corpus to read
	} else if err != nil {
		return nil, fmt.Errorf("testing: reading seed corpus from testdata: %v", err)
	}
	var corpus []CorpusEntry
	for _, file := range files {
		// TODO(jayconrod,katiehockman): determine when a file is a fuzzing input
		// based on its name. We should only read files created by writeToCorpus.
		// If we read ALL files, we won't be able to change the file format by
		// changing the extension. We also won't be able to add files like
		// README.txt explaining why the directory exists.
		if file.IsDir() {
			continue
		}
		bytes, err := ioutil.ReadFile(filepath.Join(dir, file.Name()))
		if err != nil {
			return nil, fmt.Errorf("testing: failed to read corpus file: %v", err)
		}
		corpus = append(corpus, CorpusEntry{Name: file.Name(), Data: bytes})
	}
	return corpus, nil
}

// writeToCorpus atomically writes the given bytes to a new file in testdata.
// If the directory does not exist, it will create one. If the file already
// exists, writeToCorpus will not rewrite it. writeToCorpus returns the
// file's name, or an error if it failed.
func writeToCorpus(b []byte, dir string) (name string, err error) {
	sum := fmt.Sprintf("%x", sha256.Sum256(b))
	name = filepath.Join(dir, sum)
	if err := os.MkdirAll(dir, 0777); err != nil {
		return "", err
	}
	if err := ioutil.WriteFile(name, b, 0666); err != nil {
		os.Remove(name) // remove partially written file
		return "", err
	}
	return name, nil
}
