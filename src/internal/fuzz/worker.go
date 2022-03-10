// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"sync"
	"time"
)

const (
	// workerFuzzDuration is the amount of time a worker can spend testing random
	// variations of an input given by the coordinator.
	workerFuzzDuration = 100 * time.Millisecond

	// workerTimeoutDuration is the amount of time a worker can go without
	// responding to the coordinator before being stopped.
	workerTimeoutDuration = 1 * time.Second

	// workerExitCode is used as an exit code by fuzz worker processes after an internal error.
	// This distinguishes internal errors from uncontrolled panics and other crashes.
	// Keep in sync with internal/fuzz.workerExitCode.
	workerExitCode = 70

	// workerSharedMemSize is the maximum size of the shared memory file used to
	// communicate with workers. This limits the size of fuzz inputs.
	workerSharedMemSize = 100 << 20 // 100 MB
)

// worker manages a worker process running a test binary. The worker object
// exists only in the coordinator (the process started by 'go test -fuzz').
// workerClient is used by the coordinator to send RPCs to the worker process,
// which handles them with workerServer.
type worker struct {
	dir     string   // working directory, same as package directory
	binPath string   // path to test executable
	args    []string // arguments for test executable
	env     []string // environment for test executable

	coordinator *coordinator

	memMu chan *sharedMem // mutex guarding shared memory with worker; persists across processes.

	cmd         *exec.Cmd     // current worker process
	client      *workerClient // used to communicate with worker process
	waitErr     error         // last error returned by wait, set before termC is closed.
	interrupted bool          // true after stop interrupts a running worker.
	termC       chan struct{} // closed by wait when worker process terminates
}

func newWorker(c *coordinator, dir, binPath string, args, env []string) (*worker, error) {
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

// cleanup releases persistent resources associated with the worker.
func (w *worker) cleanup() error {
	mem := <-w.memMu
	if mem == nil {
		return nil
	}
	close(w.memMu)
	return mem.Close()
}

// coordinate runs the test binary to perform fuzzing.
//
// coordinate loops until ctx is cancelled or a fatal error is encountered.
// If a test process terminates unexpectedly while fuzzing, coordinate will
// attempt to restart and continue unless the termination can be attributed
// to an interruption (from a timer or the user).
//
// While looping, coordinate receives inputs from the coordinator, passes
// those inputs to the worker process, then passes the results back to
// the coordinator.
func (w *worker) coordinate(ctx context.Context) error {
	// Main event loop.
	for {
		// Start or restart the worker if it's not running.
		if !w.isRunning() {
			if err := w.startAndPing(ctx); err != nil {
				return err
			}
		}

		select {
		case <-ctx.Done():
			// Worker was told to stop.
			err := w.stop()
			if err != nil && !w.interrupted && !isInterruptError(err) {
				return err
			}
			return ctx.Err()

		case <-w.termC:
			// Worker process terminated unexpectedly while waiting for input.
			err := w.stop()
			if w.interrupted {
				panic("worker interrupted after unexpected termination")
			}
			if err == nil || isInterruptError(err) {
				// Worker stopped, either by exiting with status 0 or after being
				// interrupted with a signal that was not sent by the coordinator.
				//
				// When the user presses ^C, on POSIX platforms, SIGINT is delivered to
				// all processes in the group concurrently, and the worker may see it
				// before the coordinator. The worker should exit 0 gracefully (in
				// theory).
				//
				// This condition is probably intended by the user, so suppress
				// the error.
				return nil
			}
			if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == workerExitCode {
				// Worker exited with a code indicating F.Fuzz was not called correctly,
				// for example, F.Fail was called first.
				return fmt.Errorf("fuzzing process exited unexpectedly due to an internal failure: %w", err)
			}
			// Worker exited non-zero or was terminated by a non-interrupt
			// signal (for example, SIGSEGV) while fuzzing.
			return fmt.Errorf("fuzzing process hung or terminated unexpectedly: %w", err)
			// TODO(jayconrod,katiehockman): if -keepfuzzing, restart worker.

		case input := <-w.coordinator.inputC:
			// Received input from coordinator.
			args := fuzzArgs{
				Limit:        input.limit,
				Timeout:      input.timeout,
				Warmup:       input.warmup,
				CoverageData: input.coverageData,
			}
			entry, resp, isInternalError, err := w.client.fuzz(ctx, input.entry, args)
			canMinimize := true
			if err != nil {
				// Error communicating with worker.
				w.stop()
				if ctx.Err() != nil {
					// Timeout or interruption.
					return ctx.Err()
				}
				if w.interrupted {
					// Communication error before we stopped the worker.
					// Report an error, but don't record a crasher.
					return fmt.Errorf("communicating with fuzzing process: %v", err)
				}
				if sig, ok := terminationSignal(w.waitErr); ok && !isCrashSignal(sig) {
					// Worker terminated by a signal that probably wasn't caused by a
					// specific input to the fuzz function. For example, on Linux,
					// the kernel (OOM killer) may send SIGKILL to a process using a lot
					// of memory. Or the shell might send SIGHUP when the terminal
					// is closed. Don't record a crasher.
					return fmt.Errorf("fuzzing process terminated by unexpected signal; no crash will be recorded: %v", w.waitErr)
				}
				if isInternalError {
					// An internal error occurred which shouldn't be considered
					// a crash.
					return err
				}
				// Unexpected termination. Set error message and fall through.
				// We'll restart the worker on the next iteration.
				// Don't attempt to minimize this since it crashed the worker.
				resp.Err = fmt.Sprintf("fuzzing process hung or terminated unexpectedly: %v", w.waitErr)
				canMinimize = false
			}
			result := fuzzResult{
				limit:         input.limit,
				count:         resp.Count,
				totalDuration: resp.TotalDuration,
				entryDuration: resp.InterestingDuration,
				entry:         entry,
				crasherMsg:    resp.Err,
				coverageData:  resp.CoverageData,
				canMinimize:   canMinimize,
			}
			w.coordinator.resultC <- result

		case input := <-w.coordinator.minimizeC:
			// Received input to minimize from coordinator.
			result, err := w.minimize(ctx, input)
			if err != nil {
				// Error minimizing. Send back the original input. If it didn't cause
				// an error before, report it as causing an error now.
				// TODO: double-check this is handled correctly when
				// implementing -keepfuzzing.
				result = fuzzResult{
					entry:       input.entry,
					crasherMsg:  input.crasherMsg,
					canMinimize: false,
					limit:       input.limit,
				}
				if result.crasherMsg == "" {
					result.crasherMsg = err.Error()
				}
			}
			w.coordinator.resultC <- result
		}
	}
}

// minimize tells a worker process to attempt to find a smaller value that
// either causes an error (if we started minimizing because we found an input
// that causes an error) or preserves new coverage (if we started minimizing
// because we found an input that expands coverage).
func (w *worker) minimize(ctx context.Context, input fuzzMinimizeInput) (min fuzzResult, err error) {
	if w.coordinator.opts.MinimizeTimeout != 0 {
		var cancel func()
		ctx, cancel = context.WithTimeout(ctx, w.coordinator.opts.MinimizeTimeout)
		defer cancel()
	}

	args := minimizeArgs{
		Limit:        input.limit,
		Timeout:      input.timeout,
		KeepCoverage: input.keepCoverage,
	}
	entry, resp, err := w.client.minimize(ctx, input.entry, args)
	if err != nil {
		// Error communicating with worker.
		w.stop()
		if ctx.Err() != nil || w.interrupted || isInterruptError(w.waitErr) {
			// Worker was interrupted, possibly by the user pressing ^C.
			// Normally, workers can handle interrupts and timeouts gracefully and
			// will return without error. An error here indicates the worker
			// may not have been in a good state, but the error won't be meaningful
			// to the user. Just return the original crasher without logging anything.
			return fuzzResult{
				entry:        input.entry,
				crasherMsg:   input.crasherMsg,
				coverageData: input.keepCoverage,
				canMinimize:  false,
				limit:        input.limit,
			}, nil
		}
		return fuzzResult{
			entry:         entry,
			crasherMsg:    fmt.Sprintf("fuzzing process hung or terminated unexpectedly while minimizing: %v", err),
			canMinimize:   false,
			limit:         input.limit,
			count:         resp.Count,
			totalDuration: resp.Duration,
		}, nil
	}

	if input.crasherMsg != "" && resp.Err == "" {
		return fuzzResult{}, fmt.Errorf("attempted to minimize a crash but could not reproduce")
	}

	return fuzzResult{
		entry:         entry,
		crasherMsg:    resp.Err,
		coverageData:  resp.CoverageData,
		canMinimize:   false,
		limit:         input.limit,
		count:         resp.Count,
		totalDuration: resp.Duration,
	}, nil
}

func (w *worker) isRunning() bool {
	return w.cmd != nil
}

// startAndPing starts the worker process and sends it a message to make sure it
// can communicate.
//
// startAndPing returns an error if any part of this didn't work, including if
// the context is expired or the worker process was interrupted before it
// responded. Errors that happen after start but before the ping response
// likely indicate that the worker did not call F.Fuzz or called F.Fail first.
// We don't record crashers for these errors.
func (w *worker) startAndPing(ctx context.Context) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if err := w.start(); err != nil {
		return err
	}
	if err := w.client.ping(ctx); err != nil {
		w.stop()
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if isInterruptError(err) {
			// User may have pressed ^C before worker responded.
			return err
		}
		// TODO: record and return stderr.
		return fmt.Errorf("fuzzing process terminated without fuzzing: %w", err)
	}
	return nil
}

// start runs a new worker process.
//
// If the process couldn't be started, start returns an error. Start won't
// return later termination errors from the process if they occur.
//
// If the process starts successfully, start returns nil. stop must be called
// once later to clean up, even if the process terminates on its own.
//
// When the process terminates, w.waitErr is set to the error (if any), and
// w.termC is closed.
func (w *worker) start() (err error) {
	if w.isRunning() {
		panic("worker already started")
	}
	w.waitErr = nil
	w.interrupted = false
	w.termC = nil

	cmd := exec.Command(w.binPath, w.args...)
	cmd.Dir = w.dir
	cmd.Env = w.env[:len(w.env):len(w.env)] // copy on append to ensure workers don't overwrite each other.

	// Create the "fuzz_in" and "fuzz_out" pipes so we can communicate with
	// the worker. We don't use stdin and stdout, since the test binary may
	// do something else with those.
	//
	// Each pipe has a reader and a writer. The coordinator writes to fuzzInW
	// and reads from fuzzOutR. The worker inherits fuzzInR and fuzzOutW.
	// The coordinator closes fuzzInR and fuzzOutW after starting the worker,
	// since we have no further need of them.
	fuzzInR, fuzzInW, err := os.Pipe()
	if err != nil {
		return err
	}
	defer fuzzInR.Close()
	fuzzOutR, fuzzOutW, err := os.Pipe()
	if err != nil {
		fuzzInW.Close()
		return err
	}
	defer fuzzOutW.Close()
	setWorkerComm(cmd, workerComm{fuzzIn: fuzzInR, fuzzOut: fuzzOutW, memMu: w.memMu})

	// Start the worker process.
	if err := cmd.Start(); err != nil {
		fuzzInW.Close()
		fuzzOutR.Close()
		return err
	}

	// Worker started successfully.
	// After this, w.client owns fuzzInW and fuzzOutR, so w.client.Close must be
	// called later by stop.
	w.cmd = cmd
	w.termC = make(chan struct{})
	comm := workerComm{fuzzIn: fuzzInW, fuzzOut: fuzzOutR, memMu: w.memMu}
	m := newMutator()
	w.client = newWorkerClient(comm, m)

	go func() {
		w.waitErr = w.cmd.Wait()
		close(w.termC)
	}()

	return nil
}

// stop tells the worker process to exit by closing w.client, then blocks until
// it terminates. If the worker doesn't terminate after a short time, stop
// signals it with os.Interrupt (where supported), then os.Kill.
//
// stop returns the error the process terminated with, if any (same as
// w.waitErr).
//
// stop must be called at least once after start returns successfully, even if
// the worker process terminates unexpectedly.
func (w *worker) stop() error {
	if w.termC == nil {
		panic("worker was not started successfully")
	}
	select {
	case <-w.termC:
		// Worker already terminated.
		if w.client == nil {
			// stop already called.
			return w.waitErr
		}
		// Possible unexpected termination.
		w.client.Close()
		w.cmd = nil
		w.client = nil
		return w.waitErr
	default:
		// Worker still running.
	}

	// Tell the worker to stop by closing fuzz_in. It won't actually stop until it
	// finishes with earlier calls.
	closeC := make(chan struct{})
	go func() {
		w.client.Close()
		close(closeC)
	}()

	sig := os.Interrupt
	if runtime.GOOS == "windows" {
		// Per https://golang.org/pkg/os/#Signal, “Interrupt is not implemented on
		// Windows; using it with os.Process.Signal will return an error.”
		// Fall back to Kill instead.
		sig = os.Kill
	}

	t := time.NewTimer(workerTimeoutDuration)
	for {
		select {
		case <-w.termC:
			// Worker terminated.
			t.Stop()
			<-closeC
			w.cmd = nil
			w.client = nil
			return w.waitErr

		case <-t.C:
			// Timer fired before worker terminated.
			w.interrupted = true
			switch sig {
			case os.Interrupt:
				// Try to stop the worker with SIGINT and wait a little longer.
				w.cmd.Process.Signal(sig)
				sig = os.Kill
				t.Reset(workerTimeoutDuration)

			case os.Kill:
				// Try to stop the worker with SIGKILL and keep waiting.
				w.cmd.Process.Signal(sig)
				sig = nil
				t.Reset(workerTimeoutDuration)

			case nil:
				// Still waiting. Print a message to let the user know why.
				fmt.Fprintf(w.coordinator.opts.Log, "waiting for fuzzing process to terminate...\n")
			}
		}
	}
}

// RunFuzzWorker is called in a worker process to communicate with the
// coordinator process in order to fuzz random inputs. RunFuzzWorker loops
// until the coordinator tells it to stop.
//
// fn is a wrapper on the fuzz function. It may return an error to indicate
// a given input "crashed". The coordinator will also record a crasher if
// the function times out or terminates the process.
//
// RunFuzzWorker returns an error if it could not communicate with the
// coordinator process.
func RunFuzzWorker(ctx context.Context, fn func(CorpusEntry) error) error {
	comm, err := getWorkerComm()
	if err != nil {
		return err
	}
	srv := &workerServer{
		workerComm: comm,
		fuzzFn: func(e CorpusEntry) (time.Duration, error) {
			timer := time.AfterFunc(10*time.Second, func() {
				panic("deadlocked!") // this error message won't be printed
			})
			defer timer.Stop()
			start := time.Now()
			err := fn(e)
			return time.Since(start), err
		},
		m: newMutator(),
	}
	return srv.serve(ctx)
}

// call is serialized and sent from the coordinator on fuzz_in. It acts as
// a minimalist RPC mechanism. Exactly one of its fields must be set to indicate
// which method to call.
type call struct {
	Ping     *pingArgs
	Fuzz     *fuzzArgs
	Minimize *minimizeArgs
}

// minimizeArgs contains arguments to workerServer.minimize. The value to
// minimize is already in shared memory.
type minimizeArgs struct {
	// Timeout is the time to spend minimizing. This may include time to start up,
	// especially if the input causes the worker process to terminated, requiring
	// repeated restarts.
	Timeout time.Duration

	// Limit is the maximum number of values to test, without spending more time
	// than Duration. 0 indicates no limit.
	Limit int64

	// KeepCoverage is a set of coverage counters the worker should attempt to
	// keep in minimized values. When provided, the worker will reject inputs that
	// don't cause at least one of these bits to be set.
	KeepCoverage []byte

	// Index is the index of the fuzz target parameter to be minimized.
	Index int
}

// minimizeResponse contains results from workerServer.minimize.
type minimizeResponse struct {
	// WroteToMem is true if the worker found a smaller input and wrote it to
	// shared memory. If minimizeArgs.KeepCoverage was set, the minimized input
	// preserved at least one coverage bit and did not cause an error.
	// Otherwise, the minimized input caused some error, recorded in Err.
	WroteToMem bool

	// Err is the error string caused by the value in shared memory, if any.
	Err string

	// CoverageData is the set of coverage bits activated by the minimized value
	// in shared memory. When set, it contains at least one bit from KeepCoverage.
	// CoverageData will be nil if Err is set or if minimization failed.
	CoverageData []byte

	// Duration is the time spent minimizing, not including starting or cleaning up.
	Duration time.Duration

	// Count is the number of values tested.
	Count int64
}

// fuzzArgs contains arguments to workerServer.fuzz. The value to fuzz is
// passed in shared memory.
type fuzzArgs struct {
	// Timeout is the time to spend fuzzing, not including starting or
	// cleaning up.
	Timeout time.Duration

	// Limit is the maximum number of values to test, without spending more time
	// than Duration. 0 indicates no limit.
	Limit int64

	// Warmup indicates whether this is part of a warmup run, meaning that
	// fuzzing should not occur. If coverageEnabled is true, then coverage data
	// should be reported.
	Warmup bool

	// CoverageData is the coverage data. If set, the worker should update its
	// local coverage data prior to fuzzing.
	CoverageData []byte
}

// fuzzResponse contains results from workerServer.fuzz.
type fuzzResponse struct {
	// Duration is the time spent fuzzing, not including starting or cleaning up.
	TotalDuration       time.Duration
	InterestingDuration time.Duration

	// Count is the number of values tested.
	Count int64

	// CoverageData is set if the value in shared memory expands coverage
	// and therefore may be interesting to the coordinator.
	CoverageData []byte

	// Err is the error string caused by the value in shared memory, which is
	// non-empty if the value in shared memory caused a crash.
	Err string

	// InternalErr is the error string caused by an internal error in the
	// worker. This shouldn't be considered a crasher.
	InternalErr string
}

// pingArgs contains arguments to workerServer.ping.
type pingArgs struct{}

// pingResponse contains results from workerServer.ping.
type pingResponse struct{}

// workerComm holds pipes and shared memory used for communication
// between the coordinator process (client) and a worker process (server).
// These values are unique to each worker; they are shared only with the
// coordinator, not with other workers.
//
// Access to shared memory is synchronized implicitly over the RPC protocol
// implemented in workerServer and workerClient. During a call, the client
// (worker) has exclusive access to shared memory; at other times, the server
// (coordinator) has exclusive access.
type workerComm struct {
	fuzzIn, fuzzOut *os.File
	memMu           chan *sharedMem // mutex guarding shared memory
}

// workerServer is a minimalist RPC server, run by fuzz worker processes.
// It allows the coordinator process (using workerClient) to call methods in a
// worker process. This system allows the coordinator to run multiple worker
// processes in parallel and to collect inputs that caused crashes from shared
// memory after a worker process terminates unexpectedly.
type workerServer struct {
	workerComm
	m *mutator

	// coverageMask is the local coverage data for the worker. It is
	// periodically updated to reflect the data in the coordinator when new
	// coverage is found.
	coverageMask []byte

	// fuzzFn runs the worker's fuzz target on the given input and returns an
	// error if it finds a crasher (the process may also exit or crash), and the
	// time it took to run the input. It sets a deadline of 10 seconds, at which
	// point it will panic with the assumption that the process is hanging or
	// deadlocked.
	fuzzFn func(CorpusEntry) (time.Duration, error)
}

// serve reads serialized RPC messages on fuzzIn. When serve receives a message,
// it calls the corresponding method, then sends the serialized result back
// on fuzzOut.
//
// serve handles RPC calls synchronously; it will not attempt to read a message
// until the previous call has finished.
//
// serve returns errors that occurred when communicating over pipes. serve
// does not return errors from method calls; those are passed through serialized
// responses.
func (ws *workerServer) serve(ctx context.Context) error {
	enc := json.NewEncoder(ws.fuzzOut)
	dec := json.NewDecoder(&contextReader{ctx: ctx, r: ws.fuzzIn})
	for {
		var c call
		if err := dec.Decode(&c); err != nil {
			if err == io.EOF || err == ctx.Err() {
				return nil
			} else {
				return err
			}
		}

		var resp any
		switch {
		case c.Fuzz != nil:
			resp = ws.fuzz(ctx, *c.Fuzz)
		case c.Minimize != nil:
			resp = ws.minimize(ctx, *c.Minimize)
		case c.Ping != nil:
			resp = ws.ping(ctx, *c.Ping)
		default:
			return errors.New("no arguments provided for any call")
		}

		if err := enc.Encode(resp); err != nil {
			return err
		}
	}
}

// chainedMutations is how many mutations are applied before the worker
// resets the input to it's original state.
// NOTE: this number was picked without much thought. It is low enough that
// it seems to create a significant diversity in mutated inputs. We may want
// to consider looking into this more closely once we have a proper performance
// testing framework. Another option is to randomly pick the number of chained
// mutations on each invocation of the workerServer.fuzz method (this appears to
// be what libFuzzer does, although there seems to be no documentation which
// explains why this choice was made.)
const chainedMutations = 5

// fuzz runs the test function on random variations of the input value in shared
// memory for a limited duration or number of iterations.
//
// fuzz returns early if it finds an input that crashes the fuzz function (with
// fuzzResponse.Err set) or an input that expands coverage (with
// fuzzResponse.InterestingDuration set).
//
// fuzz does not modify the input in shared memory. Instead, it saves the
// initial PRNG state in shared memory and increments a counter in shared
// memory before each call to the test function. The caller may reconstruct
// the crashing input with this information, since the PRNG is deterministic.
func (ws *workerServer) fuzz(ctx context.Context, args fuzzArgs) (resp fuzzResponse) {
	if args.CoverageData != nil {
		if ws.coverageMask != nil && len(args.CoverageData) != len(ws.coverageMask) {
			resp.InternalErr = fmt.Sprintf("unexpected size for CoverageData: got %d, expected %d", len(args.CoverageData), len(ws.coverageMask))
			return resp
		}
		ws.coverageMask = args.CoverageData
	}
	start := time.Now()
	defer func() { resp.TotalDuration = time.Since(start) }()

	if args.Timeout != 0 {
		var cancel func()
		ctx, cancel = context.WithTimeout(ctx, args.Timeout)
		defer cancel()
	}
	mem := <-ws.memMu
	ws.m.r.save(&mem.header().randState, &mem.header().randInc)
	defer func() {
		resp.Count = mem.header().count
		ws.memMu <- mem
	}()
	if args.Limit > 0 && mem.header().count >= args.Limit {
		resp.InternalErr = fmt.Sprintf("mem.header().count %d already exceeds args.Limit %d", mem.header().count, args.Limit)
		return resp
	}

	originalVals, err := unmarshalCorpusFile(mem.valueCopy())
	if err != nil {
		resp.InternalErr = err.Error()
		return resp
	}
	vals := make([]any, len(originalVals))
	copy(vals, originalVals)

	shouldStop := func() bool {
		return args.Limit > 0 && mem.header().count >= args.Limit
	}
	fuzzOnce := func(entry CorpusEntry) (dur time.Duration, cov []byte, errMsg string) {
		mem.header().count++
		var err error
		dur, err = ws.fuzzFn(entry)
		if err != nil {
			errMsg = err.Error()
			if errMsg == "" {
				errMsg = "fuzz function failed with no input"
			}
			return dur, nil, errMsg
		}
		if ws.coverageMask != nil && countNewCoverageBits(ws.coverageMask, coverageSnapshot) > 0 {
			return dur, coverageSnapshot, ""
		}
		return dur, nil, ""
	}

	if args.Warmup {
		dur, _, errMsg := fuzzOnce(CorpusEntry{Values: vals})
		if errMsg != "" {
			resp.Err = errMsg
			return resp
		}
		resp.InterestingDuration = dur
		if coverageEnabled {
			resp.CoverageData = coverageSnapshot
		}
		return resp
	}

	for {
		select {
		case <-ctx.Done():
			return resp
		default:
			if mem.header().count%chainedMutations == 0 {
				copy(vals, originalVals)
				ws.m.r.save(&mem.header().randState, &mem.header().randInc)
			}
			ws.m.mutate(vals, cap(mem.valueRef()))

			entry := CorpusEntry{Values: vals}
			dur, cov, errMsg := fuzzOnce(entry)
			if errMsg != "" {
				resp.Err = errMsg
				return resp
			}
			if cov != nil {
				resp.CoverageData = cov
				resp.InterestingDuration = dur
				return resp
			}
			if shouldStop() {
				return resp
			}
		}
	}
}

func (ws *workerServer) minimize(ctx context.Context, args minimizeArgs) (resp minimizeResponse) {
	start := time.Now()
	defer func() { resp.Duration = time.Now().Sub(start) }()
	mem := <-ws.memMu
	defer func() { ws.memMu <- mem }()
	vals, err := unmarshalCorpusFile(mem.valueCopy())
	if err != nil {
		panic(err)
	}
	inpHash := sha256.Sum256(mem.valueCopy())
	if args.Timeout != 0 {
		var cancel func()
		ctx, cancel = context.WithTimeout(ctx, args.Timeout)
		defer cancel()
	}

	// Minimize the values in vals, then write to shared memory. We only write
	// to shared memory after completing minimization.
	success, err := ws.minimizeInput(ctx, vals, mem, args)
	if success {
		writeToMem(vals, mem)
		outHash := sha256.Sum256(mem.valueCopy())
		mem.header().rawInMem = false
		resp.WroteToMem = true
		if err != nil {
			resp.Err = err.Error()
		} else {
			// If the values didn't change during minimization then coverageSnapshot is likely
			// a dirty snapshot which represents the very last step of minimization, not the
			// coverage for the initial input. In that case just return the coverage we were
			// given initially, since it more accurately represents the coverage map for the
			// input we are returning.
			if outHash != inpHash {
				resp.CoverageData = coverageSnapshot
			} else {
				resp.CoverageData = args.KeepCoverage
			}
		}
	}
	return resp
}

// minimizeInput applies a series of minimizing transformations on the provided
// vals, ensuring that each minimization still causes an error, or keeps
// coverage, in fuzzFn. It uses the context to determine how long to run,
// stopping once closed. It returns a bool indicating whether minimization was
// successful and an error if one was found.
func (ws *workerServer) minimizeInput(ctx context.Context, vals []any, mem *sharedMem, args minimizeArgs) (success bool, retErr error) {
	keepCoverage := args.KeepCoverage
	memBytes := mem.valueRef()
	bPtr := &memBytes
	count := &mem.header().count
	shouldStop := func() bool {
		return ctx.Err() != nil ||
			(args.Limit > 0 && *count >= args.Limit)
	}
	if shouldStop() {
		return false, nil
	}

	// Check that the original value preserves coverage or causes an error.
	// If not, then whatever caused us to think the value was interesting may
	// have been a flake, and we can't minimize it.
	*count++
	_, retErr = ws.fuzzFn(CorpusEntry{Values: vals})
	if keepCoverage != nil {
		if !hasCoverageBit(keepCoverage, coverageSnapshot) || retErr != nil {
			return false, nil
		}
	} else if retErr == nil {
		return false, nil
	}
	mem.header().rawInMem = true

	// tryMinimized runs the fuzz function with candidate replacing the value
	// at index valI. tryMinimized returns whether the input with candidate is
	// interesting for the same reason as the original input: it returns
	// an error if one was expected, or it preserves coverage.
	tryMinimized := func(candidate []byte) bool {
		prev := vals[args.Index]
		switch prev.(type) {
		case []byte:
			vals[args.Index] = candidate
		case string:
			vals[args.Index] = string(candidate)
		default:
			panic("impossible")
		}
		copy(*bPtr, candidate)
		*bPtr = (*bPtr)[:len(candidate)]
		mem.setValueLen(len(candidate))
		*count++
		_, err := ws.fuzzFn(CorpusEntry{Values: vals})
		if err != nil {
			retErr = err
			if keepCoverage != nil {
				// Now that we've found a crash, that's more important than any
				// minimization of interesting inputs that was being done. Clear out
				// keepCoverage to only minimize the crash going forward.
				keepCoverage = nil
			}
			return true
		}
		// Minimization should preserve coverage bits.
		if keepCoverage != nil && isCoverageSubset(keepCoverage, coverageSnapshot) {
			return true
		}
		vals[args.Index] = prev
		return false
	}
	switch v := vals[args.Index].(type) {
	case string:
		minimizeBytes([]byte(v), tryMinimized, shouldStop)
	case []byte:
		minimizeBytes(v, tryMinimized, shouldStop)
	default:
		panic("impossible")
	}
	return true, retErr
}

func writeToMem(vals []any, mem *sharedMem) {
	b := marshalCorpusFile(vals...)
	mem.setValue(b)
}

// ping does nothing. The coordinator calls this method to ensure the worker
// has called F.Fuzz and can communicate.
func (ws *workerServer) ping(ctx context.Context, args pingArgs) pingResponse {
	return pingResponse{}
}

// workerClient is a minimalist RPC client. The coordinator process uses a
// workerClient to call methods in each worker process (handled by
// workerServer).
type workerClient struct {
	workerComm
	m *mutator

	// mu is the mutex protecting the workerComm.fuzzIn pipe. This must be
	// locked before making calls to the workerServer. It prevents
	// workerClient.Close from closing fuzzIn while workerClient methods are
	// writing to it concurrently, and prevents multiple callers from writing to
	// fuzzIn concurrently.
	mu sync.Mutex
}

func newWorkerClient(comm workerComm, m *mutator) *workerClient {
	return &workerClient{workerComm: comm, m: m}
}

// Close shuts down the connection to the RPC server (the worker process) by
// closing fuzz_in. Close drains fuzz_out (avoiding a SIGPIPE in the worker),
// and closes it after the worker process closes the other end.
func (wc *workerClient) Close() error {
	wc.mu.Lock()
	defer wc.mu.Unlock()

	// Close fuzzIn. This signals to the server that there are no more calls,
	// and it should exit.
	if err := wc.fuzzIn.Close(); err != nil {
		wc.fuzzOut.Close()
		return err
	}

	// Drain fuzzOut and close it. When the server exits, the kernel will close
	// its end of fuzzOut, and we'll get EOF.
	if _, err := io.Copy(ioutil.Discard, wc.fuzzOut); err != nil {
		wc.fuzzOut.Close()
		return err
	}
	return wc.fuzzOut.Close()
}

// errSharedMemClosed is returned by workerClient methods that cannot access
// shared memory because it was closed and unmapped by another goroutine. That
// can happen when worker.cleanup is called in the worker goroutine while a
// workerClient.fuzz call runs concurrently.
//
// This error should not be reported. It indicates the operation was
// interrupted.
var errSharedMemClosed = errors.New("internal error: shared memory was closed and unmapped")

// minimize tells the worker to call the minimize method. See
// workerServer.minimize.
func (wc *workerClient) minimize(ctx context.Context, entryIn CorpusEntry, args minimizeArgs) (entryOut CorpusEntry, resp minimizeResponse, retErr error) {
	wc.mu.Lock()
	defer wc.mu.Unlock()

	mem, ok := <-wc.memMu
	if !ok {
		return CorpusEntry{}, minimizeResponse{}, errSharedMemClosed
	}
	mem.header().count = 0
	inp, err := corpusEntryData(entryIn)
	if err != nil {
		return CorpusEntry{}, minimizeResponse{}, err
	}
	mem.setValue(inp)
	defer func() { wc.memMu <- mem }()
	entryOut = entryIn
	entryOut.Values, err = unmarshalCorpusFile(inp)
	if err != nil {
		return CorpusEntry{}, minimizeResponse{}, fmt.Errorf("workerClient.minimize unmarshaling provided value: %v", err)
	}
	for i, v := range entryOut.Values {
		if !isMinimizable(reflect.TypeOf(v)) {
			continue
		}

		wc.memMu <- mem
		args.Index = i
		c := call{Minimize: &args}
		callErr := wc.callLocked(ctx, c, &resp)
		mem, ok = <-wc.memMu
		if !ok {
			return CorpusEntry{}, minimizeResponse{}, errSharedMemClosed
		}

		if callErr != nil {
			retErr = callErr
			if !mem.header().rawInMem {
				// An unrecoverable error occurred before minimization began.
				return entryIn, minimizeResponse{}, retErr
			}
			// An unrecoverable error occurred during minimization. mem now
			// holds the raw, unmarshalled bytes of entryIn.Values[i] that
			// caused the error.
			switch entryOut.Values[i].(type) {
			case string:
				entryOut.Values[i] = string(mem.valueCopy())
			case []byte:
				entryOut.Values[i] = mem.valueCopy()
			default:
				panic("impossible")
			}
			entryOut.Data = marshalCorpusFile(entryOut.Values...)
			// Stop minimizing; another unrecoverable error is likely to occur.
			break
		}

		if resp.WroteToMem {
			// Minimization succeeded, and mem holds the marshaled data.
			entryOut.Data = mem.valueCopy()
			entryOut.Values, err = unmarshalCorpusFile(entryOut.Data)
			if err != nil {
				return CorpusEntry{}, minimizeResponse{}, fmt.Errorf("workerClient.minimize unmarshaling minimized value: %v", err)
			}
		}

		// Prepare for next iteration of the loop.
		if args.Timeout != 0 {
			args.Timeout -= resp.Duration
			if args.Timeout <= 0 {
				break
			}
		}
		if args.Limit != 0 {
			args.Limit -= mem.header().count
			if args.Limit <= 0 {
				break
			}
		}
	}
	resp.Count = mem.header().count
	h := sha256.Sum256(entryOut.Data)
	entryOut.Path = fmt.Sprintf("%x", h[:4])
	return entryOut, resp, retErr
}

// fuzz tells the worker to call the fuzz method. See workerServer.fuzz.
func (wc *workerClient) fuzz(ctx context.Context, entryIn CorpusEntry, args fuzzArgs) (entryOut CorpusEntry, resp fuzzResponse, isInternalError bool, err error) {
	wc.mu.Lock()
	defer wc.mu.Unlock()

	mem, ok := <-wc.memMu
	if !ok {
		return CorpusEntry{}, fuzzResponse{}, true, errSharedMemClosed
	}
	mem.header().count = 0
	inp, err := corpusEntryData(entryIn)
	if err != nil {
		return CorpusEntry{}, fuzzResponse{}, true, err
	}
	mem.setValue(inp)
	wc.memMu <- mem

	c := call{Fuzz: &args}
	callErr := wc.callLocked(ctx, c, &resp)
	if resp.InternalErr != "" {
		return CorpusEntry{}, fuzzResponse{}, true, errors.New(resp.InternalErr)
	}
	mem, ok = <-wc.memMu
	if !ok {
		return CorpusEntry{}, fuzzResponse{}, true, errSharedMemClosed
	}
	defer func() { wc.memMu <- mem }()
	resp.Count = mem.header().count

	if !bytes.Equal(inp, mem.valueRef()) {
		return CorpusEntry{}, fuzzResponse{}, true, errors.New("workerServer.fuzz modified input")
	}
	needEntryOut := callErr != nil || resp.Err != "" ||
		(!args.Warmup && resp.CoverageData != nil)
	if needEntryOut {
		valuesOut, err := unmarshalCorpusFile(inp)
		if err != nil {
			return CorpusEntry{}, fuzzResponse{}, true, fmt.Errorf("unmarshaling fuzz input value after call: %v", err)
		}
		wc.m.r.restore(mem.header().randState, mem.header().randInc)
		if !args.Warmup {
			// Only mutate the valuesOut if fuzzing actually occurred.
			numMutations := ((resp.Count - 1) % chainedMutations) + 1
			for i := int64(0); i < numMutations; i++ {
				wc.m.mutate(valuesOut, cap(mem.valueRef()))
			}
		}
		dataOut := marshalCorpusFile(valuesOut...)

		h := sha256.Sum256(dataOut)
		name := fmt.Sprintf("%x", h[:4])
		entryOut = CorpusEntry{
			Parent:     entryIn.Path,
			Path:       name,
			Data:       dataOut,
			Generation: entryIn.Generation + 1,
		}
		if args.Warmup {
			// The bytes weren't mutated, so if entryIn was a seed corpus value,
			// then entryOut is too.
			entryOut.IsSeed = entryIn.IsSeed
		}
	}

	return entryOut, resp, false, callErr
}

// ping tells the worker to call the ping method. See workerServer.ping.
func (wc *workerClient) ping(ctx context.Context) error {
	wc.mu.Lock()
	defer wc.mu.Unlock()
	c := call{Ping: &pingArgs{}}
	var resp pingResponse
	return wc.callLocked(ctx, c, &resp)
}

// callLocked sends an RPC from the coordinator to the worker process and waits
// for the response. The callLocked may be cancelled with ctx.
func (wc *workerClient) callLocked(ctx context.Context, c call, resp any) (err error) {
	enc := json.NewEncoder(wc.fuzzIn)
	dec := json.NewDecoder(&contextReader{ctx: ctx, r: wc.fuzzOut})
	if err := enc.Encode(c); err != nil {
		return err
	}
	return dec.Decode(resp)
}

// contextReader wraps a Reader with a Context. If the context is cancelled
// while the underlying reader is blocked, Read returns immediately.
//
// This is useful for reading from a pipe. Closing a pipe file descriptor does
// not unblock pending Reads on that file descriptor. All copies of the pipe's
// other file descriptor (the write end) must be closed in all processes that
// inherit it. This is difficult to do correctly in the situation we care about
// (process group termination).
type contextReader struct {
	ctx context.Context
	r   io.Reader
}

func (cr *contextReader) Read(b []byte) (int, error) {
	if ctxErr := cr.ctx.Err(); ctxErr != nil {
		return 0, ctxErr
	}
	done := make(chan struct{})

	// This goroutine may stay blocked after Read returns because the underlying
	// read is blocked.
	var n int
	var err error
	go func() {
		n, err = cr.r.Read(b)
		close(done)
	}()

	select {
	case <-cr.ctx.Done():
		return 0, cr.ctx.Err()
	case <-done:
		return n, err
	}
}
