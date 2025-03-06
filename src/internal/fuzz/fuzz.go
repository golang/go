// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fuzz provides common fuzzing functionality for tests built with
// "go test" and for programs that use fuzzing functionality in the testing
// package.
package fuzz

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"internal/godebug"
	"io"
	"math/bits"
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

	// Limit is the number of random values to generate and test. If zero,
	// there will be no limit on the number of generated values.
	Limit int64

	// MinimizeTimeout is the amount of wall clock time to spend minimizing
	// after discovering a crasher. If zero, there will be no time limit. If
	// MinimizeTimeout and MinimizeLimit are both zero, then minimization will
	// be disabled.
	MinimizeTimeout time.Duration

	// MinimizeLimit is the maximum number of calls to the fuzz function to be
	// made while minimizing after finding a crash. If zero, there will be no
	// limit. Calls to the fuzz function made when minimizing also count toward
	// Limit. If MinimizeTimeout and MinimizeLimit are both zero, then
	// minimization will be disabled.
	MinimizeLimit int64

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
	if opts.Limit > 0 && int64(opts.Parallel) > opts.Limit {
		// Don't start more workers than we need.
		opts.Parallel = int(opts.Limit)
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

	// fuzzCtx is used to stop workers, for example, after finding a crasher.
	fuzzCtx, cancelWorkers := context.WithCancel(ctx)
	defer cancelWorkers()
	doneC := ctx.Done()

	// stop is called when a worker encounters a fatal error.
	var fuzzErr error
	stopping := false
	stop := func(err error) {
		if shouldPrintDebugInfo() {
			_, file, line, ok := runtime.Caller(1)
			if ok {
				c.debugLogf("stop called at %s:%d. stopping: %t", file, line, stopping)
			} else {
				c.debugLogf("stop called at unknown. stopping: %t", stopping)
			}
		}

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
	}

	// Ensure that any crash we find is written to the corpus, even if an error
	// or interruption occurs while minimizing it.
	crashWritten := false
	defer func() {
		if c.crashMinimizing == nil || crashWritten {
			return
		}
		werr := writeToCorpus(&c.crashMinimizing.entry, opts.CorpusDir)
		if werr != nil {
			err = fmt.Errorf("%w\n%v", err, werr)
			return
		}
		if err == nil {
			err = &crashError{
				path: c.crashMinimizing.entry.Path,
				err:  errors.New(c.crashMinimizing.crasherMsg),
			}
		}
	}()

	// Start workers.
	// TODO(jayconrod): do we want to support fuzzing different binaries?
	dir := "" // same as self
	binPath := os.Args[0]
	args := append([]string{"-test.fuzzworker"}, os.Args[1:]...)
	env := os.Environ() // same as self

	errC := make(chan error)
	workers := make([]*worker, opts.Parallel)
	for i := range workers {
		var err error
		workers[i], err = newWorker(c, dir, binPath, args, env)
		if err != nil {
			return err
		}
	}
	for i := range workers {
		w := workers[i]
		go func() {
			err := w.coordinate(fuzzCtx)
			if fuzzCtx.Err() != nil || isInterruptError(err) {
				err = nil
			}
			cleanErr := w.cleanup()
			if err == nil {
				err = cleanErr
			}
			errC <- err
		}()
	}

	// Main event loop.
	// Do not return until all workers have terminated. We avoid a deadlock by
	// receiving messages from workers even after ctx is canceled.
	activeWorkers := len(workers)
	statTicker := time.NewTicker(3 * time.Second)
	defer statTicker.Stop()
	defer c.logStats()

	c.logStats()
	for {
		// If there is an execution limit, and we've reached it, stop.
		if c.opts.Limit > 0 && c.count >= c.opts.Limit {
			stop(nil)
		}

		var inputC chan fuzzInput
		input, ok := c.peekInput()
		if ok && c.crashMinimizing == nil && !stopping {
			inputC = c.inputC
		}

		var minimizeC chan fuzzMinimizeInput
		minimizeInput, ok := c.peekMinimizeInput()
		if ok && !stopping {
			minimizeC = c.minimizeC
		}

		select {
		case <-doneC:
			// Interrupted, canceled, or timed out.
			// stop sets doneC to nil, so we don't busy wait here.
			stop(ctx.Err())

		case err := <-errC:
			// A worker terminated, possibly after encountering a fatal error.
			stop(err)
			activeWorkers--
			if activeWorkers == 0 {
				return fuzzErr
			}

		case result := <-c.resultC:
			// Received response from worker.
			if stopping {
				break
			}
			c.updateStats(result)

			if result.crasherMsg != "" {
				if c.warmupRun() && result.entry.IsSeed {
					target := filepath.Base(c.opts.CorpusDir)
					fmt.Fprintf(c.opts.Log, "failure while testing seed corpus entry: %s/%s\n", target, testName(result.entry.Parent))
					stop(errors.New(result.crasherMsg))
					break
				}
				if c.canMinimize() && result.canMinimize {
					if c.crashMinimizing != nil {
						// This crash is not minimized, and another crash is being minimized.
						// Ignore this one and wait for the other one to finish.
						if shouldPrintDebugInfo() {
							c.debugLogf("found unminimized crasher, skipping in favor of minimizable crasher")
						}
						break
					}
					// Found a crasher but haven't yet attempted to minimize it.
					// Send it back to a worker for minimization. Disable inputC so
					// other workers don't continue fuzzing.
					c.crashMinimizing = &result
					fmt.Fprintf(c.opts.Log, "fuzz: minimizing %d-byte failing input file\n", len(result.entry.Data))
					c.queueForMinimization(result, nil)
				} else if !crashWritten {
					// Found a crasher that's either minimized or not minimizable.
					// Write to corpus and stop.
					err := writeToCorpus(&result.entry, opts.CorpusDir)
					if err == nil {
						crashWritten = true
						err = &crashError{
							path: result.entry.Path,
							err:  errors.New(result.crasherMsg),
						}
					}
					if shouldPrintDebugInfo() {
						c.debugLogf(
							"found crasher, id: %s, parent: %s, gen: %d, size: %d, exec time: %s",
							result.entry.Path,
							result.entry.Parent,
							result.entry.Generation,
							len(result.entry.Data),
							result.entryDuration,
						)
					}
					stop(err)
				}
			} else if result.coverageData != nil {
				if c.warmupRun() {
					if shouldPrintDebugInfo() {
						c.debugLogf(
							"processed an initial input, id: %s, new bits: %d, size: %d, exec time: %s",
							result.entry.Parent,
							countBits(diffCoverage(c.coverageMask, result.coverageData)),
							len(result.entry.Data),
							result.entryDuration,
						)
					}
					c.updateCoverage(result.coverageData)
					c.warmupInputLeft--
					if c.warmupInputLeft == 0 {
						fmt.Fprintf(c.opts.Log, "fuzz: elapsed: %s, gathering baseline coverage: %d/%d completed, now fuzzing with %d workers\n", c.elapsed(), c.warmupInputCount, c.warmupInputCount, c.opts.Parallel)
						if shouldPrintDebugInfo() {
							c.debugLogf(
								"finished processing input corpus, entries: %d, initial coverage bits: %d",
								len(c.corpus.entries),
								countBits(c.coverageMask),
							)
						}
					}
				} else if keepCoverage := diffCoverage(c.coverageMask, result.coverageData); keepCoverage != nil {
					// Found a value that expanded coverage.
					// It's not a crasher, but we may want to add it to the on-disk
					// corpus and prioritize it for future fuzzing.
					// TODO(jayconrod, katiehockman): Prioritize fuzzing these
					// values which expanded coverage, perhaps based on the
					// number of new edges that this result expanded.
					// TODO(jayconrod, katiehockman): Don't write a value that's already
					// in the corpus.
					if c.canMinimize() && result.canMinimize && c.crashMinimizing == nil {
						// Send back to workers to find a smaller value that preserves
						// at least one new coverage bit.
						c.queueForMinimization(result, keepCoverage)
					} else {
						// Update the coordinator's coverage mask and save the value.
						inputSize := len(result.entry.Data)
						entryNew, err := c.addCorpusEntries(true, result.entry)
						if err != nil {
							stop(err)
							break
						}
						if !entryNew {
							if shouldPrintDebugInfo() {
								c.debugLogf(
									"ignoring duplicate input which increased coverage, id: %s",
									result.entry.Path,
								)
							}
							break
						}
						c.updateCoverage(keepCoverage)
						c.inputQueue.enqueue(result.entry)
						c.interestingCount++
						if shouldPrintDebugInfo() {
							c.debugLogf(
								"new interesting input, id: %s, parent: %s, gen: %d, new bits: %d, total bits: %d, size: %d, exec time: %s",
								result.entry.Path,
								result.entry.Parent,
								result.entry.Generation,
								countBits(keepCoverage),
								countBits(c.coverageMask),
								inputSize,
								result.entryDuration,
							)
						}
					}
				} else {
					if shouldPrintDebugInfo() {
						c.debugLogf(
							"worker reported interesting input that doesn't expand coverage, id: %s, parent: %s, canMinimize: %t",
							result.entry.Path,
							result.entry.Parent,
							result.canMinimize,
						)
					}
				}
			} else if c.warmupRun() {
				// No error or coverage data was reported for this input during
				// warmup, so continue processing results.
				c.warmupInputLeft--
				if c.warmupInputLeft == 0 {
					fmt.Fprintf(c.opts.Log, "fuzz: elapsed: %s, testing seed corpus: %d/%d completed, now fuzzing with %d workers\n", c.elapsed(), c.warmupInputCount, c.warmupInputCount, c.opts.Parallel)
					if shouldPrintDebugInfo() {
						c.debugLogf(
							"finished testing-only phase, entries: %d",
							len(c.corpus.entries),
						)
					}
				}
			}

		case inputC <- input:
			// Sent the next input to a worker.
			c.sentInput(input)

		case minimizeC <- minimizeInput:
			// Sent the next input for minimization to a worker.
			c.sentMinimizeInput(minimizeInput)

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
	path string
	err  error
}

func (e *crashError) Error() string {
	return e.err.Error()
}

func (e *crashError) Unwrap() error {
	return e.err
}

func (e *crashError) CrashPath() string {
	return e.path
}

type corpus struct {
	entries []CorpusEntry
	hashes  map[[sha256.Size]byte]bool
}

// addCorpusEntries adds entries to the corpus, and optionally writes the entries
// to the cache directory. If an entry is already in the corpus it is skipped. If
// all of the entries are unique, addCorpusEntries returns true and a nil error,
// if at least one of the entries was a duplicate, it returns false and a nil error.
func (c *coordinator) addCorpusEntries(addToCache bool, entries ...CorpusEntry) (bool, error) {
	noDupes := true
	for _, e := range entries {
		data, err := corpusEntryData(e)
		if err != nil {
			return false, err
		}
		h := sha256.Sum256(data)
		if c.corpus.hashes[h] {
			noDupes = false
			continue
		}
		if addToCache {
			if err := writeToCorpus(&e, c.opts.CacheDir); err != nil {
				return false, err
			}
			// For entries written to disk, we don't hold onto the bytes,
			// since the corpus would consume a significant amount of
			// memory.
			e.Data = nil
		}
		c.corpus.hashes[h] = true
		c.corpus.entries = append(c.corpus.entries, e)
	}
	return noDupes, nil
}

// CorpusEntry represents an individual input for fuzzing.
//
// We must use an equivalent type in the testing and testing/internal/testdeps
// packages, but testing can't import this package directly, and we don't want
// to export this type from testing. Instead, we use the same struct type and
// use a type alias (not a defined type) for convenience.
type CorpusEntry = struct {
	Parent string

	// Path is the path of the corpus file, if the entry was loaded from disk.
	// For other entries, including seed values provided by f.Add, Path is the
	// name of the test, e.g. seed#0 or its hash.
	Path string

	// Data is the raw input data. Data should only be populated for seed
	// values. For on-disk corpus files, Data will be nil, as it will be loaded
	// from disk using Path.
	Data []byte

	// Values is the unmarshaled values from a corpus file.
	Values []any

	Generation int

	// IsSeed indicates whether this entry is part of the seed corpus.
	IsSeed bool
}

// corpusEntryData returns the raw input bytes, either from the data struct
// field, or from disk.
func corpusEntryData(ce CorpusEntry) ([]byte, error) {
	if ce.Data != nil {
		return ce.Data, nil
	}

	return os.ReadFile(ce.Path)
}

type fuzzInput struct {
	// entry is the value to test initially. The worker will randomly mutate
	// values from this starting point.
	entry CorpusEntry

	// timeout is the time to spend fuzzing variations of this input,
	// not including starting or cleaning up.
	timeout time.Duration

	// limit is the maximum number of calls to the fuzz function the worker may
	// make. The worker may make fewer calls, for example, if it finds an
	// error early. If limit is zero, there is no limit on calls to the
	// fuzz function.
	limit int64

	// warmup indicates whether this is a warmup input before fuzzing begins. If
	// true, the input should not be fuzzed.
	warmup bool

	// coverageData reflects the coordinator's current coverageMask.
	coverageData []byte
}

type fuzzResult struct {
	// entry is an interesting value or a crasher.
	entry CorpusEntry

	// crasherMsg is an error message from a crash. It's "" if no crash was found.
	crasherMsg string

	// canMinimize is true if the worker should attempt to minimize this result.
	// It may be false because an attempt has already been made.
	canMinimize bool

	// coverageData is set if the worker found new coverage.
	coverageData []byte

	// limit is the number of values the coordinator asked the worker
	// to test. 0 if there was no limit.
	limit int64

	// count is the number of values the worker actually tested.
	count int64

	// totalDuration is the time the worker spent testing inputs.
	totalDuration time.Duration

	// entryDuration is the time the worker spent execution an interesting result
	entryDuration time.Duration
}

type fuzzMinimizeInput struct {
	// entry is an interesting value or crasher to minimize.
	entry CorpusEntry

	// crasherMsg is an error message from a crash. It's "" if no crash was found.
	// If set, the worker will attempt to find a smaller input that also produces
	// an error, though not necessarily the same error.
	crasherMsg string

	// limit is the maximum number of calls to the fuzz function the worker may
	// make. The worker may make fewer calls, for example, if it can't reproduce
	// an error. If limit is zero, there is no limit on calls to the fuzz function.
	limit int64

	// timeout is the time to spend minimizing this input.
	// A zero timeout means no limit.
	timeout time.Duration

	// keepCoverage is a set of coverage bits that entry found that were not in
	// the coordinator's combined set. When minimizing, the worker should find an
	// input that preserves at least one of these bits. keepCoverage is nil for
	// crashing inputs.
	keepCoverage []byte
}

// coordinator holds channels that workers can use to communicate with
// the coordinator.
type coordinator struct {
	opts CoordinateFuzzingOpts

	// startTime is the time we started the workers after loading the corpus.
	// Used for logging.
	startTime time.Time

	// inputC is sent values to fuzz by the coordinator. Any worker may receive
	// values from this channel. Workers send results to resultC.
	inputC chan fuzzInput

	// minimizeC is sent values to minimize by the coordinator. Any worker may
	// receive values from this channel. Workers send results to resultC.
	minimizeC chan fuzzMinimizeInput

	// resultC is sent results of fuzzing by workers. The coordinator
	// receives these. Multiple types of messages are allowed.
	resultC chan fuzzResult

	// count is the number of values fuzzed so far.
	count int64

	// countLastLog is the number of values fuzzed when the output was last
	// logged.
	countLastLog int64

	// timeLastLog is the time at which the output was last logged.
	timeLastLog time.Time

	// interestingCount is the number of unique interesting values which have
	// been found this execution.
	interestingCount int

	// warmupInputCount is the count of all entries in the corpus which will
	// need to be received from workers to run once during warmup, but not fuzz.
	// This could be for coverage data, or only for the purposes of verifying
	// that the seed corpus doesn't have any crashers. See warmupRun.
	warmupInputCount int

	// warmupInputLeft is the number of entries in the corpus which still need
	// to be received from workers to run once during warmup, but not fuzz.
	// See warmupInputLeft.
	warmupInputLeft int

	// duration is the time spent fuzzing inside workers, not counting time
	// starting up or tearing down.
	duration time.Duration

	// countWaiting is the number of fuzzing executions the coordinator is
	// waiting on workers to complete.
	countWaiting int64

	// corpus is a set of interesting values, including the seed corpus and
	// generated values that workers reported as interesting.
	corpus corpus

	// minimizationAllowed is true if one or more of the types of fuzz
	// function's parameters can be minimized.
	minimizationAllowed bool

	// inputQueue is a queue of inputs that workers should try fuzzing. This is
	// initially populated from the seed corpus and cached inputs. More inputs
	// may be added as new coverage is discovered.
	inputQueue queue

	// minimizeQueue is a queue of inputs that caused errors or exposed new
	// coverage. Workers should attempt to find smaller inputs that do the
	// same thing.
	minimizeQueue queue

	// crashMinimizing is the crash that is currently being minimized.
	crashMinimizing *fuzzResult

	// coverageMask aggregates coverage that was found for all inputs in the
	// corpus. Each byte represents a single basic execution block. Each set bit
	// within the byte indicates that an input has triggered that block at least
	// 1 << n times, where n is the position of the bit in the byte. For example, a
	// value of 12 indicates that separate inputs have triggered this block
	// between 4-7 times and 8-15 times.
	coverageMask []byte
}

func newCoordinator(opts CoordinateFuzzingOpts) (*coordinator, error) {
	// Make sure all the seed corpus has marshaled data.
	for i := range opts.Seed {
		if opts.Seed[i].Data == nil && opts.Seed[i].Values != nil {
			opts.Seed[i].Data = marshalCorpusFile(opts.Seed[i].Values...)
		}
	}
	c := &coordinator{
		opts:        opts,
		startTime:   time.Now(),
		inputC:      make(chan fuzzInput),
		minimizeC:   make(chan fuzzMinimizeInput),
		resultC:     make(chan fuzzResult),
		timeLastLog: time.Now(),
		corpus:      corpus{hashes: make(map[[sha256.Size]byte]bool)},
	}
	if err := c.readCache(); err != nil {
		return nil, err
	}
	if opts.MinimizeLimit > 0 || opts.MinimizeTimeout > 0 {
		for _, t := range opts.Types {
			if isMinimizable(t) {
				c.minimizationAllowed = true
				break
			}
		}
	}

	covSize := len(coverage())
	if covSize == 0 {
		fmt.Fprintf(c.opts.Log, "warning: the test binary was not built with coverage instrumentation, so fuzzing will run without coverage guidance and may be inefficient\n")
		// Even though a coverage-only run won't occur, we should still run all
		// of the seed corpus to make sure there are no existing failures before
		// we start fuzzing.
		c.warmupInputCount = len(c.opts.Seed)
		for _, e := range c.opts.Seed {
			c.inputQueue.enqueue(e)
		}
	} else {
		c.warmupInputCount = len(c.corpus.entries)
		for _, e := range c.corpus.entries {
			c.inputQueue.enqueue(e)
		}
		// Set c.coverageMask to a clean []byte full of zeros.
		c.coverageMask = make([]byte, covSize)
	}
	c.warmupInputLeft = c.warmupInputCount

	if len(c.corpus.entries) == 0 {
		fmt.Fprintf(c.opts.Log, "warning: starting with empty corpus\n")
		var vals []any
		for _, t := range opts.Types {
			vals = append(vals, zeroValue(t))
		}
		data := marshalCorpusFile(vals...)
		h := sha256.Sum256(data)
		name := fmt.Sprintf("%x", h[:4])
		c.addCorpusEntries(false, CorpusEntry{Path: name, Data: data})
	}

	return c, nil
}

func (c *coordinator) updateStats(result fuzzResult) {
	c.count += result.count
	c.countWaiting -= result.limit
	c.duration += result.totalDuration
}

func (c *coordinator) logStats() {
	now := time.Now()
	if c.warmupRun() {
		runSoFar := c.warmupInputCount - c.warmupInputLeft
		if coverageEnabled {
			fmt.Fprintf(c.opts.Log, "fuzz: elapsed: %s, gathering baseline coverage: %d/%d completed\n", c.elapsed(), runSoFar, c.warmupInputCount)
		} else {
			fmt.Fprintf(c.opts.Log, "fuzz: elapsed: %s, testing seed corpus: %d/%d completed\n", c.elapsed(), runSoFar, c.warmupInputCount)
		}
	} else if c.crashMinimizing != nil {
		fmt.Fprintf(c.opts.Log, "fuzz: elapsed: %s, minimizing\n", c.elapsed())
	} else {
		rate := float64(c.count-c.countLastLog) / now.Sub(c.timeLastLog).Seconds()
		if coverageEnabled {
			total := c.warmupInputCount + c.interestingCount
			fmt.Fprintf(c.opts.Log, "fuzz: elapsed: %s, execs: %d (%.0f/sec), new interesting: %d (total: %d)\n", c.elapsed(), c.count, rate, c.interestingCount, total)
		} else {
			fmt.Fprintf(c.opts.Log, "fuzz: elapsed: %s, execs: %d (%.0f/sec)\n", c.elapsed(), c.count, rate)
		}
	}
	c.countLastLog = c.count
	c.timeLastLog = now
}

// peekInput returns the next value that should be sent to workers.
// If the number of executions is limited, the returned value includes
// a limit for one worker. If there are no executions left, peekInput returns
// a zero value and false.
//
// peekInput doesn't actually remove the input from the queue. The caller
// must call sentInput after sending the input.
//
// If the input queue is empty and the coverage/testing-only run has completed,
// queue refills it from the corpus.
func (c *coordinator) peekInput() (fuzzInput, bool) {
	if c.opts.Limit > 0 && c.count+c.countWaiting >= c.opts.Limit {
		// Already making the maximum number of calls to the fuzz function.
		// Don't send more inputs right now.
		return fuzzInput{}, false
	}
	if c.inputQueue.len == 0 {
		if c.warmupRun() {
			// Wait for coverage/testing-only run to finish before sending more
			// inputs.
			return fuzzInput{}, false
		}
		c.refillInputQueue()
	}

	entry, ok := c.inputQueue.peek()
	if !ok {
		panic("input queue empty after refill")
	}
	input := fuzzInput{
		entry:   entry.(CorpusEntry),
		timeout: workerFuzzDuration,
		warmup:  c.warmupRun(),
	}
	if c.coverageMask != nil {
		input.coverageData = bytes.Clone(c.coverageMask)
	}
	if input.warmup {
		// No fuzzing will occur, but it should count toward the limit set by
		// -fuzztime.
		input.limit = 1
		return input, true
	}

	if c.opts.Limit > 0 {
		input.limit = c.opts.Limit / int64(c.opts.Parallel)
		if c.opts.Limit%int64(c.opts.Parallel) > 0 {
			input.limit++
		}
		remaining := c.opts.Limit - c.count - c.countWaiting
		if input.limit > remaining {
			input.limit = remaining
		}
	}
	return input, true
}

// sentInput updates internal counters after an input is sent to c.inputC.
func (c *coordinator) sentInput(input fuzzInput) {
	c.inputQueue.dequeue()
	c.countWaiting += input.limit
}

// refillInputQueue refills the input queue from the corpus after it becomes
// empty.
func (c *coordinator) refillInputQueue() {
	for _, e := range c.corpus.entries {
		c.inputQueue.enqueue(e)
	}
}

// queueForMinimization creates a fuzzMinimizeInput from result and adds it
// to the minimization queue to be sent to workers.
func (c *coordinator) queueForMinimization(result fuzzResult, keepCoverage []byte) {
	if shouldPrintDebugInfo() {
		c.debugLogf(
			"queueing input for minimization, id: %s, parent: %s, keepCoverage: %t, crasher: %t",
			result.entry.Path,
			result.entry.Parent,
			keepCoverage != nil,
			result.crasherMsg != "",
		)
	}
	if result.crasherMsg != "" {
		c.minimizeQueue.clear()
	}

	input := fuzzMinimizeInput{
		entry:        result.entry,
		crasherMsg:   result.crasherMsg,
		keepCoverage: keepCoverage,
	}
	c.minimizeQueue.enqueue(input)
}

// peekMinimizeInput returns the next input that should be sent to workers for
// minimization.
func (c *coordinator) peekMinimizeInput() (fuzzMinimizeInput, bool) {
	if !c.canMinimize() {
		// Already making the maximum number of calls to the fuzz function.
		// Don't send more inputs right now.
		return fuzzMinimizeInput{}, false
	}
	v, ok := c.minimizeQueue.peek()
	if !ok {
		return fuzzMinimizeInput{}, false
	}
	input := v.(fuzzMinimizeInput)

	if c.opts.MinimizeTimeout > 0 {
		input.timeout = c.opts.MinimizeTimeout
	}
	if c.opts.MinimizeLimit > 0 {
		input.limit = c.opts.MinimizeLimit
	} else if c.opts.Limit > 0 {
		if input.crasherMsg != "" {
			input.limit = c.opts.Limit
		} else {
			input.limit = c.opts.Limit / int64(c.opts.Parallel)
			if c.opts.Limit%int64(c.opts.Parallel) > 0 {
				input.limit++
			}
		}
	}
	if c.opts.Limit > 0 {
		remaining := c.opts.Limit - c.count - c.countWaiting
		if input.limit > remaining {
			input.limit = remaining
		}
	}
	return input, true
}

// sentMinimizeInput removes an input from the minimization queue after it's
// sent to minimizeC.
func (c *coordinator) sentMinimizeInput(input fuzzMinimizeInput) {
	c.minimizeQueue.dequeue()
	c.countWaiting += input.limit
}

// warmupRun returns true while the coordinator is running inputs without
// mutating them as a warmup before fuzzing. This could be to gather baseline
// coverage data for entries in the corpus, or to test all of the seed corpus
// for errors before fuzzing begins.
//
// The coordinator doesn't store coverage data in the cache with each input
// because that data would be invalid when counter offsets in the test binary
// change.
//
// When gathering coverage, the coordinator sends each entry to a worker to
// gather coverage for that entry only, without fuzzing or minimizing. This
// phase ends when all workers have finished, and the coordinator has a combined
// coverage map.
func (c *coordinator) warmupRun() bool {
	return c.warmupInputLeft > 0
}

// updateCoverage sets bits in c.coverageMask that are set in newCoverage.
// updateCoverage returns the number of newly set bits. See the comment on
// coverageMask for the format.
func (c *coordinator) updateCoverage(newCoverage []byte) int {
	if len(newCoverage) != len(c.coverageMask) {
		panic(fmt.Sprintf("number of coverage counters changed at runtime: %d, expected %d", len(newCoverage), len(c.coverageMask)))
	}
	newBitCount := 0
	for i := range newCoverage {
		diff := newCoverage[i] &^ c.coverageMask[i]
		newBitCount += bits.OnesCount8(diff)
		c.coverageMask[i] |= newCoverage[i]
	}
	return newBitCount
}

// canMinimize returns whether the coordinator should attempt to find smaller
// inputs that reproduce a crash or new coverage.
func (c *coordinator) canMinimize() bool {
	return c.minimizationAllowed &&
		(c.opts.Limit == 0 || c.count+c.countWaiting < c.opts.Limit)
}

func (c *coordinator) elapsed() time.Duration {
	return time.Since(c.startTime).Round(1 * time.Second)
}

// readCache creates a combined corpus from seed values and values in the cache
// (in GOCACHE/fuzz).
//
// TODO(fuzzing): need a mechanism that can remove values that
// aren't useful anymore, for example, because they have the wrong type.
func (c *coordinator) readCache() error {
	if _, err := c.addCorpusEntries(false, c.opts.Seed...); err != nil {
		return err
	}
	entries, err := ReadCorpus(c.opts.CacheDir, c.opts.Types)
	if err != nil {
		if _, ok := err.(*MalformedCorpusError); !ok {
			// It's okay if some files in the cache directory are malformed and
			// are not included in the corpus, but fail if it's an I/O error.
			return err
		}
		// TODO(jayconrod,katiehockman): consider printing some kind of warning
		// indicating the number of files which were skipped because they are
		// malformed.
	}
	if _, err := c.addCorpusEntries(false, entries...); err != nil {
		return err
	}
	return nil
}

// MalformedCorpusError is an error found while reading the corpus from the
// filesystem. All of the errors are stored in the errs list. The testing
// framework uses this to report malformed files in testdata.
type MalformedCorpusError struct {
	errs []error
}

func (e *MalformedCorpusError) Error() string {
	msgs := make([]string, len(e.errs))
	for i, s := range e.errs {
		msgs[i] = s.Error()
	}
	return strings.Join(msgs, "\n")
}

// ReadCorpus reads the corpus from the provided dir. The returned corpus
// entries are guaranteed to match the given types. Any malformed files will
// be saved in a MalformedCorpusError and returned, along with the most recent
// error.
func ReadCorpus(dir string, types []reflect.Type) ([]CorpusEntry, error) {
	files, err := os.ReadDir(dir)
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
		data, err := os.ReadFile(filename)
		if err != nil {
			return nil, fmt.Errorf("failed to read corpus file: %v", err)
		}
		var vals []any
		vals, err = readCorpusData(data, types)
		if err != nil {
			errs = append(errs, fmt.Errorf("%q: %v", filename, err))
			continue
		}
		corpus = append(corpus, CorpusEntry{Path: filename, Values: vals})
	}
	if len(errs) > 0 {
		return corpus, &MalformedCorpusError{errs: errs}
	}
	return corpus, nil
}

func readCorpusData(data []byte, types []reflect.Type) ([]any, error) {
	vals, err := unmarshalCorpusFile(data)
	if err != nil {
		return nil, fmt.Errorf("unmarshal: %v", err)
	}
	if err = CheckCorpus(vals, types); err != nil {
		return nil, err
	}
	return vals, nil
}

// CheckCorpus verifies that the types in vals match the expected types
// provided.
func CheckCorpus(vals []any, types []reflect.Type) error {
	if len(vals) != len(types) {
		return fmt.Errorf("wrong number of values in corpus entry: %d, want %d", len(vals), len(types))
	}
	valsT := make([]reflect.Type, len(vals))
	for valsI, v := range vals {
		valsT[valsI] = reflect.TypeOf(v)
	}
	for i := range types {
		if valsT[i] != types[i] {
			return fmt.Errorf("mismatched types in corpus entry: %v, want %v", valsT, types)
		}
	}
	return nil
}

// writeToCorpus atomically writes the given bytes to a new file in testdata. If
// the directory does not exist, it will create one. If the file already exists,
// writeToCorpus will not rewrite it. writeToCorpus sets entry.Path to the new
// file that was just written or an error if it failed.
func writeToCorpus(entry *CorpusEntry, dir string) (err error) {
	sum := fmt.Sprintf("%x", sha256.Sum256(entry.Data))[:16]
	entry.Path = filepath.Join(dir, sum)
	if err := os.MkdirAll(dir, 0777); err != nil {
		return err
	}
	if err := os.WriteFile(entry.Path, entry.Data, 0666); err != nil {
		os.Remove(entry.Path) // remove partially written file
		return err
	}
	return nil
}

func testName(path string) string {
	return filepath.Base(path)
}

func zeroValue(t reflect.Type) any {
	for _, v := range zeroVals {
		if reflect.TypeOf(v) == t {
			return v
		}
	}
	panic(fmt.Sprintf("unsupported type: %v", t))
}

var zeroVals []any = []any{
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

var debugInfo = godebug.New("#fuzzdebug").Value() == "1"

func shouldPrintDebugInfo() bool {
	return debugInfo
}

func (c *coordinator) debugLogf(format string, args ...any) {
	t := time.Now().Format("2006-01-02 15:04:05.999999999")
	fmt.Fprintf(c.opts.Log, t+" DEBUG "+format+"\n", args...)
}
