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
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"time"
)

// CoordinateFuzzingOpts is a set of arguments for CoordinateFuzzing.
// The zero value is valid for each field unless specified otherwise.
type CoordinateFuzzingOpts struct {
	// Log is a writer for logging progress messages and warnings.
	// If nil, io.Discard will be used instead.
	Log io.Writer

	// Timeout is the amount of wall clock time to spend fuzzing after the corpus
	// has loaded. If zero, there will be no time limit.
	Timeout time.Duration

	// Count is the number of random values to generate and test. If zero,
	// there will be no limit on the number of generated values.
	Count int64

	// parallel is the number of worker processes to run in parallel. If zero,
	// CoordinateFuzzing will run GOMAXPROCS workers.
	Parallel int

	// Seed is a list of seed values added by the fuzz target with testing.F.Add
	// and in testdata.
	Seed []CorpusEntry

	// Types is the list of types which make up a corpus entry.
	// Types must be set and must match values in Seed.
	Types []reflect.Type

	// CorpusDir is a directory where files containing values that crash the
	// code being tested may be written. CorpusDir must be set.
	CorpusDir string

	// CacheDir is a directory containing additional "interesting" values.
	// The fuzzer may derive new values from these, and may write new values here.
	CacheDir string
}

// CoordinateFuzzing creates several worker processes and communicates with
// them to test random inputs that could trigger crashes and expose bugs.
// The worker processes run the same binary in the same directory with the
// same environment variables as the coordinator process. Workers also run
// with the same arguments as the coordinator, except with the -test.fuzzworker
// flag prepended to the argument list.
//
// If a crash occurs, the function will return an error containing information
// about the crash, which can be reported to the user.
func CoordinateFuzzing(ctx context.Context, opts CoordinateFuzzingOpts) (err error) {
	if err := ctx.Err(); err != nil {
		return err
	}
	if opts.Log == nil {
		opts.Log = io.Discard
	}
	if opts.Parallel == 0 {
		opts.Parallel = runtime.GOMAXPROCS(0)
	}
	if opts.Count > 0 && int64(opts.Parallel) > opts.Count {
		// Don't start more workers than we need.
		opts.Parallel = int(opts.Count)
	}

	c, err := newCoordinator(opts)
	if err != nil {
		return err
	}

	if opts.Timeout > 0 {
		var cancel func()
		ctx, cancel = context.WithTimeout(ctx, opts.Timeout)
		defer cancel()
	}

	// TODO(jayconrod): do we want to support fuzzing different binaries?
	dir := "" // same as self
	binPath := os.Args[0]
	args := append([]string{"-test.fuzzworker"}, os.Args[1:]...)
	env := os.Environ() // same as self

	// newWorker creates a worker but doesn't start it yet.
	newWorker := func() (*worker, error) {
		mem, err := sharedMemTempFile(workerSharedMemSize)
		if err != nil {
			return nil, err
		}
		memMu := make(chan *sharedMem, 1)
		memMu <- mem
		return &worker{
			dir:         dir,
			binPath:     binPath,
			args:        args,
			env:         env[:len(env):len(env)], // copy on append to ensure workers don't overwrite each other.
			coordinator: c,
			memMu:       memMu,
		}, nil
	}

	// fuzzCtx is used to stop workers, for example, after finding a crasher.
	fuzzCtx, cancelWorkers := context.WithCancel(ctx)
	defer cancelWorkers()
	doneC := ctx.Done()
	inputC := c.inputC

	// stop is called when a worker encounters a fatal error.
	var fuzzErr error
	stopping := false
	stop := func(err error) {
		if err == fuzzCtx.Err() || isInterruptError(err) {
			// Suppress cancellation errors and terminations due to SIGINT.
			// The messages are not helpful since either the user triggered the error
			// (with ^C) or another more helpful message will be printed (a crasher).
			err = nil
		}
		if err != nil && (fuzzErr == nil || fuzzErr == ctx.Err()) {
			fuzzErr = err
		}
		if stopping {
			return
		}
		stopping = true
		cancelWorkers()
		doneC = nil
		inputC = nil
	}

	// Start workers.
	errC := make(chan error)
	workers := make([]*worker, opts.Parallel)
	for i := range workers {
		var err error
		workers[i], err = newWorker()
		if err != nil {
			return err
		}
	}
	for i := range workers {
		w := workers[i]
		go func() {
			err := w.coordinate(fuzzCtx)
			cleanErr := w.cleanup()
			if err == nil {
				err = cleanErr
			}
			errC <- err
		}()
	}

	// Main event loop.
	// Do not return until all workers have terminated. We avoid a deadlock by
	// receiving messages from workers even after ctx is cancelled.
	activeWorkers := len(workers)
	input, ok := c.nextInput()
	if !ok {
		panic("no input")
	}
	statTicker := time.NewTicker(3 * time.Second)
	defer statTicker.Stop()
	defer c.logStats()
	crashWritten := false

	for {
		select {
		case <-doneC:
			// Interrupted, cancelled, or timed out.
			// stop sets doneC to nil so we don't busy wait here.
			stop(ctx.Err())

		case result := <-c.resultC:
			// Received response from worker.
			c.updateStats(result)
			if c.opts.Count > 0 && c.count >= c.opts.Count {
				stop(nil)
			}

			if result.crasherMsg != "" {
				// Found a crasher. Write it to testdata and return it.
				if crashWritten {
					break
				}
				fileName, err := writeToCorpus(result.entry.Data, opts.CorpusDir)
				if err == nil {
					crashWritten = true
					err = &crashError{
						name: filepath.Base(fileName),
						err:  errors.New(result.crasherMsg),
					}
				}
				// TODO(jayconrod,katiehockman): if -keepfuzzing, report the error to
				// the user and restart the crashed worker.
				stop(err)
			} else if result.isInteresting {
				// Found an interesting value that expanded coverage.
				// This is not a crasher, but we should minimize it, add it to the
				// on-disk corpus, and prioritize it for future fuzzing.
				// TODO(jayconrod, katiehockman): Prioritize fuzzing these values which
				// expanded coverage.
				// TODO(jayconrod, katiehockman): Don't write a value that's already
				// in the corpus.
				c.corpus.entries = append(c.corpus.entries, result.entry)
				if opts.CacheDir != "" {
					if _, err := writeToCorpus(result.entry.Data, opts.CacheDir); err != nil {
						stop(err)
					}
				}
			}

			if inputC == nil && !stopping {
				// inputC was disabled earlier because we hit the limit on the number
				// of inputs to fuzz (nextInput returned false).
				// Workers can do less work than requested though, so we might be
				// below the limit now. Call nextInput again and re-enable inputC if so.
				if input, ok = c.nextInput(); ok {
					inputC = c.inputC
				}
			}

		case err := <-errC:
			// A worker terminated, possibly after encountering a fatal error.
			stop(err)
			activeWorkers--
			if activeWorkers == 0 {
				return fuzzErr
			}

		case inputC <- input:
			// Send the next input to any worker.
			if input, ok = c.nextInput(); !ok {
				inputC = nil
			}

		case <-statTicker.C:
			c.logStats()
		}
	}

	// TODO(jayconrod,katiehockman): if a crasher can't be written to the corpus,
	// write to the cache instead.
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

	// Values is the unmarshaled values from a corpus file.
	Values []interface{}
}

type fuzzInput struct {
	// entry is the value to test initially. The worker will randomly mutate
	// values from this starting point.
	entry CorpusEntry

	// countRequested is the number of values to test. If non-zero, the worker
	// will stop after testing this many values, if it hasn't already stopped.
	countRequested int64
}

type fuzzResult struct {
	// entry is an interesting value or a crasher.
	entry CorpusEntry

	// crasherMsg is an error message from a crash. It's "" if no crash was found.
	crasherMsg string

	// isInteresting is true if the worker found new coverage. We should minimize
	// the value, cache it, and prioritize it for further fuzzing.
	isInteresting bool

	// countRequested is the number of values the coordinator asked the worker
	// to test. 0 if there was no limit.
	countRequested int64

	// count is the number of values the worker actually tested.
	count int64

	// duration is the time the worker spent testing inputs.
	duration time.Duration
}

// coordinator holds channels that workers can use to communicate with
// the coordinator.
type coordinator struct {
	opts CoordinateFuzzingOpts

	// startTime is the time we started the workers after loading the corpus.
	// Used for logging.
	startTime time.Time

	// inputC is sent values to fuzz by the coordinator. Any worker may receive
	// values from this channel.
	inputC chan fuzzInput

	// resultC is sent results of fuzzing by workers. The coordinator
	// receives these. Multiple types of messages are allowed.
	resultC chan fuzzResult

	// count is the number of values fuzzed so far.
	count int64

	// duration is the time spent fuzzing inside workers, not counting time
	// starting up or tearing down.
	duration time.Duration

	// countWaiting is the number of values the coordinator is currently waiting
	// for workers to fuzz.
	countWaiting int64

	// corpus is a set of interesting values, including the seed corpus and
	// generated values that workers reported as interesting.
	corpus corpus

	// corpusIndex is the next value to send to workers.
	// TODO(jayconrod,katiehockman): need a scheduling algorithm that chooses
	// which corpus value to send next (or generates something new).
	corpusIndex int
}

func newCoordinator(opts CoordinateFuzzingOpts) (*coordinator, error) {
	// Make sure all of the seed corpus has marshalled data.
	for i := range opts.Seed {
		if opts.Seed[i].Data == nil {
			opts.Seed[i].Data = marshalCorpusFile(opts.Seed[i].Values...)
		}
	}
	corpus, err := readCache(opts.Seed, opts.Types, opts.CacheDir)
	if err != nil {
		return nil, err
	}
	if len(corpus.entries) == 0 {
		var vals []interface{}
		for _, t := range opts.Types {
			vals = append(vals, zeroValue(t))
		}
		corpus.entries = append(corpus.entries, CorpusEntry{Data: marshalCorpusFile(vals...), Values: vals})
	}
	c := &coordinator{
		opts:      opts,
		startTime: time.Now(),
		inputC:    make(chan fuzzInput),
		resultC:   make(chan fuzzResult),
		corpus:    corpus,
	}

	return c, nil
}

func (c *coordinator) updateStats(result fuzzResult) {
	// Adjust total stats.
	c.count += result.count
	c.countWaiting -= result.countRequested
	c.duration += result.duration
}

func (c *coordinator) logStats() {
	elapsed := time.Since(c.startTime)
	rate := float64(c.count) / elapsed.Seconds()
	fmt.Fprintf(c.opts.Log, "elapsed: %.1fs, execs: %d (%.0f/sec), workers: %d\n", elapsed.Seconds(), c.count, rate, c.opts.Parallel)
}

// nextInput returns the next value that should be sent to workers.
// If the number of executions is limited, the returned value includes
// a limit for one worker. If there are no executions left, nextInput returns
// a zero value and false.
func (c *coordinator) nextInput() (fuzzInput, bool) {
	if c.opts.Count > 0 && c.count+c.countWaiting >= c.opts.Count {
		// Workers already testing all requested inputs.
		return fuzzInput{}, false
	}

	e := c.corpus.entries[c.corpusIndex]
	c.corpusIndex = (c.corpusIndex + 1) % (len(c.corpus.entries))
	var n int64
	if c.opts.Count > 0 {
		n = c.opts.Count / int64(c.opts.Parallel)
		if c.opts.Count%int64(c.opts.Parallel) > 0 {
			n++
		}
		remaining := c.opts.Count - c.count - c.countWaiting
		if n > remaining {
			n = remaining
		}
		c.countWaiting += n
	}
	return fuzzInput{entry: e, countRequested: n}, true
}

// readCache creates a combined corpus from seed values and values in the cache
// (in GOCACHE/fuzz).
//
// TODO(jayconrod,katiehockman): need a mechanism that can remove values that
// aren't useful anymore, for example, because they have the wrong type.
func readCache(seed []CorpusEntry, types []reflect.Type, cacheDir string) (corpus, error) {
	var c corpus
	c.entries = append(c.entries, seed...)
	entries, err := ReadCorpus(cacheDir, types)
	if err != nil {
		if _, ok := err.(*MalformedCorpusError); !ok {
			// It's okay if some files in the cache directory are malformed and
			// are not included in the corpus, but fail if it's an I/O error.
			return corpus{}, err
		}
		// TODO(jayconrod,katiehockman): consider printing some kind of warning
		// indicating the number of files which were skipped because they are
		// malformed.
	}
	c.entries = append(c.entries, entries...)
	return c, nil
}

// MalformedCorpusError is an error found while reading the corpus from the
// filesystem. All of the errors are stored in the errs list. The testing
// framework uses this to report malformed files in testdata.
type MalformedCorpusError struct {
	errs []error
}

func (e *MalformedCorpusError) Error() string {
	var msgs []string
	for _, s := range e.errs {
		msgs = append(msgs, s.Error())
	}
	return strings.Join(msgs, "\n")
}

// ReadCorpus reads the corpus from the provided dir. The returned corpus
// entries are guaranteed to match the given types. Any malformed files will
// be saved in a MalformedCorpusError and returned, along with the most recent
// error.
func ReadCorpus(dir string, types []reflect.Type) ([]CorpusEntry, error) {
	files, err := ioutil.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil, nil // No corpus to read
	} else if err != nil {
		return nil, fmt.Errorf("reading seed corpus from testdata: %v", err)
	}
	var corpus []CorpusEntry
	var errs []error
	for _, file := range files {
		// TODO(jayconrod,katiehockman): determine when a file is a fuzzing input
		// based on its name. We should only read files created by writeToCorpus.
		// If we read ALL files, we won't be able to change the file format by
		// changing the extension. We also won't be able to add files like
		// README.txt explaining why the directory exists.
		if file.IsDir() {
			continue
		}
		filename := filepath.Join(dir, file.Name())
		data, err := ioutil.ReadFile(filename)
		if err != nil {
			return nil, fmt.Errorf("failed to read corpus file: %v", err)
		}
		var vals []interface{}
		vals, err = readCorpusData(data, types)
		if err != nil {
			errs = append(errs, fmt.Errorf("%q: %v", filename, err))
			continue
		}
		corpus = append(corpus, CorpusEntry{Name: filename, Data: data, Values: vals})
	}
	if len(errs) > 0 {
		return corpus, &MalformedCorpusError{errs: errs}
	}
	return corpus, nil
}

func readCorpusData(data []byte, types []reflect.Type) ([]interface{}, error) {
	vals, err := unmarshalCorpusFile(data)
	if err != nil {
		return nil, fmt.Errorf("unmarshal: %v", err)
	}
	if len(vals) != len(types) {
		return nil, fmt.Errorf("wrong number of values in corpus file: %d, want %d", len(vals), len(types))
	}
	for i := range types {
		if reflect.TypeOf(vals[i]) != types[i] {
			return nil, fmt.Errorf("mismatched types in corpus file: %v, want %v", vals, types)
		}
	}
	return vals, nil
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

func zeroValue(t reflect.Type) interface{} {
	for _, v := range zeroVals {
		if reflect.TypeOf(v) == t {
			return v
		}
	}
	panic(fmt.Sprintf("unsupported type: %v", t))
}

var zeroVals []interface{} = []interface{}{
	[]byte(""),
	string(""),
	false,
	byte(0),
	rune(0),
	float32(0),
	float64(0),
	int(0),
	int8(0),
	int16(0),
	int32(0),
	int64(0),
	uint(0),
	uint8(0),
	uint16(0),
	uint32(0),
	uint64(0),
}
