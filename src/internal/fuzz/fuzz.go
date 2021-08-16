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
	"math/bits"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
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
	// after discovering a crasher. If zero, there will be no time limit.
	MinimizeTimeout time.Duration

	// MinimizeLimit is the maximum number of calls to the fuzz function to be
	// made while minimizing after finding a crash. If zero, there will be
	// no limit. Calls to the fuzz function made when minimizing also count
	// toward Limit.
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

	// Ensure that any crash we find is written to the corpus, even if an error
	// or interruption occurs while minimizing it.
	var crashMinimizing *fuzzResult
	crashWritten := false
	defer func() {
		if crashMinimizing == nil || crashWritten {
			return
		}
		fileName, werr := writeToCorpus(crashMinimizing.entry.Data, opts.CorpusDir)
		if werr != nil {
			err = fmt.Errorf("%w\n%v", err, werr)
			return
		}
		if err == nil {
			err = &crashError{
				name: filepath.Base(fileName),
				err:  errors.New(crashMinimizing.crasherMsg),
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
	// receiving messages from workers even after ctx is cancelled.
	activeWorkers := len(workers)
	input, ok := c.nextInput()
	if !ok {
		panic("no input")
	}
	statTicker := time.NewTicker(3 * time.Second)
	defer statTicker.Stop()
	defer c.logStats()

	for {
		select {
		case <-doneC:
			// Interrupted, cancelled, or timed out.
			// stop sets doneC to nil so we don't busy wait here.
			stop(ctx.Err())

		case result := <-c.resultC:
			// Received response from worker.
			c.updateStats(result)
			if c.opts.Limit > 0 && c.count >= c.opts.Limit {
				stop(nil)
			}

			if result.crasherMsg != "" {
				if c.canMinimize() && !result.minimizeAttempted {
					if crashMinimizing != nil {
						// This crash is not minimized, and another crash is being minimized.
						// Ignore this one and wait for the other one to finish.
						break
					}
					// Found a crasher but haven't yet attempted to minimize it.
					// Send it back to a worker for minimization. Disable inputC so
					// other workers don't continue fuzzing.
					crashMinimizing = &result
					inputC = nil
					fmt.Fprintf(c.opts.Log, "found a crash, minimizing...\n")
					c.minimizeC <- c.minimizeInputForResult(result)
				} else if !crashWritten {
					// Found a crasher that's either minimized or not minimizable.
					// Write to corpus and stop.
					fileName, err := writeToCorpus(result.entry.Data, opts.CorpusDir)
					if err == nil {
						crashWritten = true
						err = &crashError{
							name: filepath.Base(fileName),
							err:  errors.New(result.crasherMsg),
						}
					}
					if printDebugInfo() {
						fmt.Fprintf(
							c.opts.Log,
							"DEBUG new crasher, elapsed: %s, id: %s, parent: %s, gen: %d, size: %d, exec time: %s\n",
							time.Since(c.startTime),
							result.entry.Name,
							result.entry.Parent,
							result.entry.Generation,
							len(result.entry.Data),
							result.entryDuration,
						)
					}
					stop(err)
				}
			} else if result.coverageData != nil {
				newBitCount := c.updateCoverage(result.coverageData)
				if newBitCount > 0 && !c.coverageOnlyRun() {
					// Found an interesting value that expanded coverage.
					// This is not a crasher, but we should add it to the
					// on-disk corpus, and prioritize it for future fuzzing.
					// TODO(jayconrod, katiehockman): Prioritize fuzzing these
					// values which expanded coverage, perhaps based on the
					// number of new edges that this result expanded.
					// TODO(jayconrod, katiehockman): Don't write a value that's already
					// in the corpus.
					c.interestingCount++
					c.corpus.entries = append(c.corpus.entries, result.entry)
					if opts.CacheDir != "" {
						if _, err := writeToCorpus(result.entry.Data, opts.CacheDir); err != nil {
							stop(err)
						}
					}
					if printDebugInfo() {
						fmt.Fprintf(
							c.opts.Log,
							"DEBUG new interesting input, elapsed: %s, id: %s, parent: %s, gen: %d, new bits: %d, total bits: %d, size: %d, exec time: %s\n",
							time.Since(c.startTime),
							result.entry.Name,
							result.entry.Parent,
							result.entry.Generation,
							newBitCount,
							countBits(c.coverageMask),
							len(result.entry.Data),
							result.entryDuration,
						)
					}
				} else if c.coverageOnlyRun() {
					c.covOnlyInputs--
					if printDebugInfo() {
						fmt.Fprintf(
							c.opts.Log,
							"DEBUG processed an initial input, elapsed: %s, id: %s, new bits: %d, size: %d, exec time: %s\n",
							time.Since(c.startTime),
							result.entry.Parent,
							newBitCount,
							len(result.entry.Data),
							result.entryDuration,
						)
					}
					if c.covOnlyInputs == 0 {
						// The coordinator has finished getting a baseline for
						// coverage. Tell all of the workers to inialize their
						// baseline coverage data (by setting interestingCount
						// to 0).
						c.interestingCount = 0
						if printDebugInfo() {
							fmt.Fprintf(
								c.opts.Log,
								"DEBUG finished processing input corpus, elapsed: %s, entries: %d, initial coverage bits: %d\n",
								time.Since(c.startTime),
								len(c.corpus.entries),
								countBits(c.coverageMask),
							)
						}
					}
				} else {
					if printDebugInfo() {
						fmt.Fprintf(
							c.opts.Log,
							"DEBUG worker reported interesting input that doesn't expand coverage, elapsed: %s, id: %s, parent: %s\n",
							time.Since(c.startTime),
							result.entry.Name,
							result.entry.Parent,
						)
					}
				}
			}
			if inputC == nil && crashMinimizing == nil && !stopping && !c.coverageOnlyRun() {
				// Re-enable inputC if it was disabled earlier because we hit the limit
				// on the number of inputs to fuzz (nextInput returned false). Workers
				// can do less work than requested, so after receiving a result above,
				// we might be below the limit now.
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
			if c.corpusIndex == 0 && c.coverageOnlyRun() {
				// The coordinator is currently trying to run all of the corpus
				// entries to gather baseline coverage data, and all of the
				// inputs have been passed to inputC. Block any more inputs from
				// being passed to the workers for now.
				inputC = nil
			} else if input, ok = c.nextInput(); !ok {
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
//
// TODO: split marshalled and unmarshalled types. In most places, we only need
// one or the other.
type CorpusEntry = struct {
	Parent string

	// Name is the name of the corpus file, if the entry was loaded from the
	// seed corpus. It can be used with -run. For entries added with f.Add and
	// entries generated by the mutator, Name is empty.
	Name string

	// Data is the raw data loaded from a corpus file.
	Data []byte

	// Values is the unmarshaled values from a corpus file.
	Values []interface{}

	Generation int
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

	// coverageOnly indicates whether this input is for a coverage-only run. If
	// true, the input should not be fuzzed.
	coverageOnly bool

	// interestingCount reflects the coordinator's current interestingCount
	// value.
	interestingCount int64

	// coverageData reflects the coordinator's current coverageData.
	coverageData []byte
}

type fuzzResult struct {
	// entry is an interesting value or a crasher.
	entry CorpusEntry

	// crasherMsg is an error message from a crash. It's "" if no crash was found.
	crasherMsg string

	// minimizeAttempted is true if the worker attempted to minimize this input.
	// The worker may or may not have succeeded.
	minimizeAttempted bool

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

	// interestingCount is the number of unique interesting values which have
	// been found this execution.
	interestingCount int64

	// covOnlyInputs is the number of entries in the corpus which still need to
	// be sent to a worker to gather baseline coverage data.
	covOnlyInputs int

	// duration is the time spent fuzzing inside workers, not counting time
	// starting up or tearing down.
	duration time.Duration

	// countWaiting is the number of fuzzing executions the coordinator is
	// waiting on workers to complete.
	countWaiting int64

	// corpus is a set of interesting values, including the seed corpus and
	// generated values that workers reported as interesting.
	corpus corpus

	// corpusIndex is the next value to send to workers.
	// TODO(jayconrod,katiehockman): need a scheduling algorithm that chooses
	// which corpus value to send next (or generates something new).
	corpusIndex int

	// typesAreMinimizable is true if one or more of the types of fuzz function's
	// parameters can be minimized.
	typesAreMinimizable bool

	// coverageMask aggregates coverage that was found for all inputs in the
	// corpus. Each byte represents a single basic execution block. Each set bit
	// within the byte indicates that an input has triggered that block at least
	// 1 << n times, where n is the position of the bit in the byte. For example, a
	// value of 12 indicates that separate inputs have triggered this block
	// between 4-7 times and 8-15 times.
	coverageMask []byte
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
	covOnlyInputs := len(corpus.entries)
	if len(corpus.entries) == 0 {
		var vals []interface{}
		for _, t := range opts.Types {
			vals = append(vals, zeroValue(t))
		}
		data := marshalCorpusFile(vals...)
		h := sha256.Sum256(data)
		name := fmt.Sprintf("%x", h[:4])
		corpus.entries = append(corpus.entries, CorpusEntry{Name: name, Data: data, Values: vals})
	}
	c := &coordinator{
		opts:          opts,
		startTime:     time.Now(),
		inputC:        make(chan fuzzInput),
		minimizeC:     make(chan fuzzMinimizeInput),
		resultC:       make(chan fuzzResult),
		corpus:        corpus,
		covOnlyInputs: covOnlyInputs,
	}
	for _, t := range opts.Types {
		if isMinimizable(t) {
			c.typesAreMinimizable = true
			break
		}
	}

	covSize := len(coverage())
	if covSize == 0 {
		fmt.Fprintf(c.opts.Log, "warning: coverage-guided fuzzing is not supported on this platform\n")
		c.covOnlyInputs = 0
	} else {
		// Set c.coverageData to a clean []byte full of zeros.
		c.coverageMask = make([]byte, covSize)
	}

	if c.covOnlyInputs > 0 {
		// Set c.interestingCount to -1 so the workers know when the coverage
		// run is finished and can update their local coverage data.
		c.interestingCount = -1
	}

	return c, nil
}

func (c *coordinator) updateStats(result fuzzResult) {
	c.count += result.count
	c.countWaiting -= result.limit
	c.duration += result.totalDuration
}

func (c *coordinator) logStats() {
	elapsed := time.Since(c.startTime)
	if c.coverageOnlyRun() {
		fmt.Fprintf(c.opts.Log, "gathering baseline coverage, elapsed: %.1fs, workers: %d, left: %d\n", elapsed.Seconds(), c.opts.Parallel, c.covOnlyInputs)
	} else {
		rate := float64(c.count) / elapsed.Seconds()
		fmt.Fprintf(c.opts.Log, "fuzzing, elapsed: %.1fs, execs: %d (%.0f/sec), workers: %d, interesting: %d\n", elapsed.Seconds(), c.count, rate, c.opts.Parallel, c.interestingCount)
	}
}

// nextInput returns the next value that should be sent to workers.
// If the number of executions is limited, the returned value includes
// a limit for one worker. If there are no executions left, nextInput returns
// a zero value and false.
func (c *coordinator) nextInput() (fuzzInput, bool) {
	if c.opts.Limit > 0 && c.count+c.countWaiting >= c.opts.Limit {
		// Workers already testing all requested inputs.
		return fuzzInput{}, false
	}
	input := fuzzInput{
		entry:            c.corpus.entries[c.corpusIndex],
		interestingCount: c.interestingCount,
		coverageData:     make([]byte, len(c.coverageMask)),
		timeout:          workerFuzzDuration,
	}
	copy(input.coverageData, c.coverageMask)
	c.corpusIndex = (c.corpusIndex + 1) % (len(c.corpus.entries))

	if c.coverageOnlyRun() {
		// This is a coverage-only run, so this input shouldn't be fuzzed,
		// and shouldn't be included in the count of generated values.
		input.coverageOnly = true
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
		c.countWaiting += input.limit
	}
	return input, true
}

// minimizeInputForResult returns an input for minimization based on the given
// fuzzing result that either caused a failure or expanded coverage.
func (c *coordinator) minimizeInputForResult(result fuzzResult) fuzzMinimizeInput {
	input := fuzzMinimizeInput{
		entry:      result.entry,
		crasherMsg: result.crasherMsg,
	}
	input.limit = 0
	if c.opts.MinimizeTimeout > 0 {
		input.timeout = c.opts.MinimizeTimeout
	}
	if c.opts.MinimizeLimit > 0 {
		input.limit = c.opts.MinimizeLimit
	} else if c.opts.Limit > 0 {
		if result.crasherMsg != "" {
			input.limit = c.opts.Limit
		} else {
			input.limit = c.opts.Limit / int64(c.opts.Parallel)
			if c.opts.Limit%int64(c.opts.Parallel) > 0 {
				input.limit++
			}
		}
	}
	remaining := c.opts.Limit - c.count - c.countWaiting
	if input.limit > remaining {
		input.limit = remaining
	}
	c.countWaiting += input.limit
	return input
}

func (c *coordinator) coverageOnlyRun() bool {
	return c.covOnlyInputs > 0
}

// updateCoverage sets bits in c.coverageData that are set in newCoverage.
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
	return c.typesAreMinimizable &&
		(c.opts.Limit == 0 || c.count+c.countWaiting < c.opts.Limit)
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
	if err = CheckCorpus(vals, types); err != nil {
		return nil, err
	}
	return vals, nil
}

// CheckCorpus verifies that the types in vals match the expected types
// provided. If not, attempt to convert them. If that's not possible, return an
// error.
func CheckCorpus(vals []interface{}, types []reflect.Type) error {
	if len(vals) != len(types) {
		return fmt.Errorf("wrong number of values in corpus file: %d, want %d", len(vals), len(types))
	}
	for i := range types {
		orig := reflect.ValueOf(vals[i])
		origType := orig.Type()
		wantType := types[i]
		if origType == wantType {
			continue // already the same type
		}
		// Attempt to convert the corpus value to the expected type
		if !origType.ConvertibleTo(wantType) {
			return fmt.Errorf("cannot convert %v to %v", origType, wantType)
		}
		convertedVal, ok := convertToType(orig, wantType)
		if !ok {
			return fmt.Errorf("error converting %v to %v", origType, wantType)
		}
		// TODO: Check that the value didn't change.
		// e.g. val went from int64(-1) -> uint(0) -> int64(0) which should fail

		// Updates vals to use the newly converted value of the expected type.
		vals[i] = convertedVal.Interface()
	}
	return nil
}

func convertToType(orig reflect.Value, t reflect.Type) (converted reflect.Value, ok bool) {
	// Convert might panic even if ConvertibleTo returns true, so catch
	// that panic and return false.
	defer func() {
		if r := recover(); r != nil {
			ok = false
		}
	}()
	return orig.Convert(t), true
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

var (
	debugInfo     bool
	debugInfoOnce sync.Once
)

func printDebugInfo() bool {
	debugInfoOnce.Do(func() {
		debug := strings.Split(os.Getenv("GODEBUG"), ",")
		for _, f := range debug {
			if f == "fuzzdebug=1" {
				debugInfo = true
				break
			}
		}
	})
	return debugInfo
}
