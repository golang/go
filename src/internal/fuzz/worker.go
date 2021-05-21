// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
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
	// interestingCount starts at -1, like the coordinator does, so that the
	// worker client's coverage data is updated after a coverage-only run.
	interestingCount := int64(-1)

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
			// Worker exited non-zero or was terminated by a non-interrupt signal
			// (for example, SIGSEGV) while fuzzing.
			return fmt.Errorf("fuzzing process terminated unexpectedly: %w", err)
			// TODO(jayconrod,katiehockman): if -keepfuzzing, restart worker.

		case input := <-w.coordinator.inputC:
			// Received input from coordinator.
			args := fuzzArgs{Limit: input.countRequested, Timeout: workerFuzzDuration, CoverageOnly: input.coverageOnly}
			if interestingCount < input.interestingCount {
				// The coordinator's coverage data has changed, so send the data
				// to the client.
				args.CoverageData = input.coverageData
			}
			value, resp, err := w.client.fuzz(ctx, input.entry.Data, args)
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
				if w.waitErr == nil || isInterruptError(w.waitErr) {
					// Worker stopped, either by exiting with status 0 or after being
					// interrupted with a signal (not sent by coordinator). See comment in
					// termC case above.
					//
					// Since we expect I/O errors around interrupts, ignore this error.
					return nil
				}
				// Unexpected termination. Set error message and fall through.
				// We'll restart the worker on the next iteration.
				resp.Err = fmt.Sprintf("fuzzing process terminated unexpectedly: %v", w.waitErr)
			}
			result := fuzzResult{
				countRequested: input.countRequested,
				count:          resp.Count,
				duration:       resp.Duration,
			}
			if resp.Err != "" {
				result.entry = CorpusEntry{Data: value}
				result.crasherMsg = resp.Err
			} else if resp.CoverageData != nil {
				result.entry = CorpusEntry{Data: value}
				result.coverageData = resp.CoverageData
			}
			w.coordinator.resultC <- result

		case crasher := <-w.coordinator.minimizeC:
			// Received input to minimize from coordinator.
			minRes, err := w.minimize(ctx, crasher)
			if err != nil {
				// Failed to minimize. Send back the original crash.
				fmt.Fprintln(w.coordinator.opts.Log, err)
				minRes = crasher
				minRes.minimized = true
			}
			w.coordinator.resultC <- minRes
		}
	}
}

// minimize tells a worker process to attempt to find a smaller value that
// causes an error. minimize may restart the worker repeatedly if the error
// causes (or already caused) the worker process to terminate.
//
// TODO: support minimizing inputs that expand coverage in a specific way,
// for example, by ensuring that an input activates a specific set of counters.
func (w *worker) minimize(ctx context.Context, input fuzzResult) (min fuzzResult, err error) {
	if w.coordinator.opts.MinimizeTimeout != 0 {
		var cancel func()
		ctx, cancel = context.WithTimeout(ctx, w.coordinator.opts.MinimizeTimeout)
		defer cancel()
	}

	min = input
	min.minimized = true

	args := minimizeArgs{
		Limit:   w.coordinator.opts.MinimizeLimit,
		Timeout: w.coordinator.opts.MinimizeTimeout,
	}
	value, resp, err := w.client.minimize(ctx, input.entry.Data, args)
	if err != nil {
		// Error communicating with worker.
		w.stop()
		if ctx.Err() != nil || w.interrupted || isInterruptError(w.waitErr) {
			// Worker was interrupted, possibly by the user pressing ^C.
			// Normally, workers can handle interrupts and timeouts gracefully and
			// will return without error. An error here indicates the worker
			// may not have been in a good state, but the error won't be meaningful
			// to the user. Just return the original crasher without logging anything.
			return min, nil
		}
		return fuzzResult{}, fmt.Errorf("fuzzing process terminated unexpectedly while minimizing: %w", w.waitErr)
	}

	if resp.Err == "" {
		// Minimization did not find a smaller input that caused a crash.
		return min, nil
	}
	min.crasherMsg = resp.Err
	min.count = resp.Count
	min.duration = resp.Duration
	min.entry.Data = value
	return min, nil
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
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

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
	w.client = newWorkerClient(workerComm{fuzzIn: fuzzInW, fuzzOut: fuzzOutR, memMu: w.memMu})

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
	srv := &workerServer{workerComm: comm, fuzzFn: fn, m: newMutator()}
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
}

// minimizeResponse contains results from workerServer.minimize.
type minimizeResponse struct {
	// Err is the error string caused by the value in shared memory.
	// If Err is empty, minimize was unable to find any shorter values that
	// caused errors, and the value in shared memory is the original value.
	Err string

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

	// CoverageOnly indicates whether this is a coverage-only run (ie. fuzzing
	// should not occur).
	CoverageOnly bool

	// CoverageData is the coverage data. If set, the worker should update its
	// local coverage data prior to fuzzing.
	CoverageData []byte
}

// fuzzResponse contains results from workerServer.fuzz.
type fuzzResponse struct {
	// Duration is the time spent fuzzing, not including starting or cleaning up.
	Duration time.Duration

	// Count is the number of values tested.
	Count int64

	// CoverageData is set if the value in shared memory expands coverage
	// and therefore may be interesting to the coordinator.
	CoverageData []byte

	// Err is the error string caused by the value in shared memory, which is
	// non-empty if the value in shared memory caused a crash.
	Err string
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

	// coverageData is the local coverage data for the worker. It is
	// periodically updated to reflect the data in the coordinator when new
	// edges are hit.
	coverageData []byte

	// fuzzFn runs the worker's fuzz function on the given input and returns
	// an error if it finds a crasher (the process may also exit or crash).
	fuzzFn func(CorpusEntry) error
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
	// This goroutine may stay blocked after serve returns because the underlying
	// read blocks, even after the file descriptor in this process is closed. The
	// pipe must be closed by the client, too.
	errC := make(chan error, 1)
	go func() {
		enc := json.NewEncoder(ws.fuzzOut)
		dec := json.NewDecoder(ws.fuzzIn)
		for {
			if ctx.Err() != nil {
				return
			}

			var c call
			if err := dec.Decode(&c); err == io.EOF {
				return
			} else if err != nil {
				errC <- err
				return
			}
			if ctx.Err() != nil {
				return
			}

			var resp interface{}
			switch {
			case c.Fuzz != nil:
				resp = ws.fuzz(ctx, *c.Fuzz)
			case c.Minimize != nil:
				resp = ws.minimize(ctx, *c.Minimize)
			case c.Ping != nil:
				resp = ws.ping(ctx, *c.Ping)
			default:
				errC <- errors.New("no arguments provided for any call")
				return
			}

			if err := enc.Encode(resp); err != nil {
				errC <- err
				return
			}
		}
	}()

	select {
	case <-ctx.Done():
		// Stop handling messages when ctx.Done() is closed. This normally happens
		// when the worker process receives a SIGINT signal, which on POSIX platforms
		// is sent to the process group when ^C is pressed.
		return ctx.Err()
	case err := <-errC:
		return err
	}
}

// fuzz runs the test function on random variations of a given input value for
// a given amount of time. fuzz returns early if it finds an input that crashes
// the fuzz function or an input that expands coverage.
func (ws *workerServer) fuzz(ctx context.Context, args fuzzArgs) (resp fuzzResponse) {
	if args.CoverageData != nil {
		ws.coverageData = args.CoverageData
	}
	start := time.Now()
	defer func() { resp.Duration = time.Since(start) }()

	fuzzCtx, cancel := context.WithTimeout(ctx, args.Timeout)
	defer cancel()
	mem := <-ws.memMu
	defer func() {
		resp.Count = mem.header().count
		ws.memMu <- mem
	}()

	vals, err := unmarshalCorpusFile(mem.valueCopy())
	if err != nil {
		panic(err)
	}

	if args.CoverageOnly {
		ws.fuzzFn(CorpusEntry{Values: vals})
		resp.CoverageData = coverageSnapshot
		return resp
	}

	if cov := coverage(); len(cov) != len(ws.coverageData) {
		panic(fmt.Sprintf("num edges changed at runtime: %d, expected %d", len(cov), len(ws.coverageData)))
	}
	for {
		select {
		case <-fuzzCtx.Done():
			return resp

		default:
			mem.header().count++
			ws.m.mutate(vals, cap(mem.valueRef()))
			writeToMem(vals, mem)
			if err := ws.fuzzFn(CorpusEntry{Values: vals}); err != nil {
				resp.Err = err.Error()
				if resp.Err == "" {
					resp.Err = "fuzz function failed with no output"
				}
				return resp
			}
			for i := range coverageSnapshot {
				if ws.coverageData[i] == 0 && coverageSnapshot[i] > ws.coverageData[i] {
					// TODO(jayconrod,katie): minimize this.
					resp.CoverageData = coverageSnapshot
					return resp
				}
			}
			if args.Limit > 0 && mem.header().count == args.Limit {
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
	if args.Timeout != 0 {
		var cancel func()
		ctx, cancel = context.WithTimeout(ctx, args.Timeout)
		defer cancel()
	}

	// Minimize the values in vals, then write to shared memory. We only write
	// to shared memory after completing minimization. If the worker terminates
	// unexpectedly before then, the coordinator will use the original input.
	err = ws.minimizeInput(ctx, vals, &mem.header().count, args.Limit)
	writeToMem(vals, mem)
	if err != nil {
		resp.Err = err.Error()
	}
	return resp
}

// minimizeInput applies a series of minimizing transformations on the provided
// vals, ensuring that each minimization still causes an error in fuzzFn. Before
// every call to fuzzFn, it marshals the new vals and writes it to the provided
// mem just in case an unrecoverable error occurs. It uses the context to
// determine how long to run, stopping once closed. It returns the last error it
// found.
func (ws *workerServer) minimizeInput(ctx context.Context, vals []interface{}, count *int64, limit int64) error {
	shouldStop := func() bool {
		return ctx.Err() != nil || (limit > 0 && *count >= limit)
	}
	if shouldStop() {
		return nil
	}

	var valI int
	var retErr error
	tryMinimized := func(candidate interface{}) bool {
		prev := vals[valI]
		// Set vals[valI] to the candidate after it has been
		// properly cast. We know that candidate must be of
		// the same type as prev, so use that as a reference.
		switch c := candidate.(type) {
		case float64:
			switch prev.(type) {
			case float32:
				vals[valI] = float32(c)
			case float64:
				vals[valI] = c
			default:
				panic("impossible")
			}
		case uint:
			switch prev.(type) {
			case uint:
				vals[valI] = c
			case uint8:
				vals[valI] = uint8(c)
			case uint16:
				vals[valI] = uint16(c)
			case uint32:
				vals[valI] = uint32(c)
			case uint64:
				vals[valI] = uint64(c)
			case int:
				vals[valI] = int(c)
			case int8:
				vals[valI] = int8(c)
			case int16:
				vals[valI] = int16(c)
			case int32:
				vals[valI] = int32(c)
			case int64:
				vals[valI] = int64(c)
			default:
				panic("impossible")
			}
		case []byte:
			switch prev.(type) {
			case []byte:
				vals[valI] = c
			case string:
				vals[valI] = string(c)
			default:
				panic("impossible")
			}
		default:
			panic("impossible")
		}
		err := ws.fuzzFn(CorpusEntry{Values: vals})
		if err != nil {
			retErr = err
			return true
		}
		*count++
		vals[valI] = prev
		return false
	}

	for valI = range vals {
		if shouldStop() {
			return retErr
		}
		switch v := vals[valI].(type) {
		case bool:
			continue // can't minimize
		case float32:
			minimizeFloat(float64(v), tryMinimized, shouldStop)
		case float64:
			minimizeFloat(v, tryMinimized, shouldStop)
		case uint:
			minimizeInteger(v, tryMinimized, shouldStop)
		case uint8:
			minimizeInteger(uint(v), tryMinimized, shouldStop)
		case uint16:
			minimizeInteger(uint(v), tryMinimized, shouldStop)
		case uint32:
			minimizeInteger(uint(v), tryMinimized, shouldStop)
		case uint64:
			if uint64(uint(v)) != v {
				// Skip minimizing a uint64 on 32 bit platforms, since we'll truncate the
				// value when casting
				continue
			}
			minimizeInteger(uint(v), tryMinimized, shouldStop)
		case int:
			minimizeInteger(uint(v), tryMinimized, shouldStop)
		case int8:
			minimizeInteger(uint(v), tryMinimized, shouldStop)
		case int16:
			minimizeInteger(uint(v), tryMinimized, shouldStop)
		case int32:
			minimizeInteger(uint(v), tryMinimized, shouldStop)
		case int64:
			if int64(int(v)) != v {
				// Skip minimizing a int64 on 32 bit platforms, since we'll truncate the
				// value when casting
				continue
			}
			minimizeInteger(uint(v), tryMinimized, shouldStop)
		case string:
			minimizeBytes([]byte(v), tryMinimized, shouldStop)
		case []byte:
			minimizeBytes(v, tryMinimized, shouldStop)
		default:
			panic("unreachable")
		}
	}
	return retErr
}

func writeToMem(vals []interface{}, mem *sharedMem) {
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

	mu  sync.Mutex
	enc *json.Encoder
	dec *json.Decoder
}

func newWorkerClient(comm workerComm) *workerClient {
	return &workerClient{
		workerComm: comm,
		enc:        json.NewEncoder(comm.fuzzIn),
		dec:        json.NewDecoder(comm.fuzzOut),
	}
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
func (wc *workerClient) minimize(ctx context.Context, valueIn []byte, args minimizeArgs) (valueOut []byte, resp minimizeResponse, err error) {
	wc.mu.Lock()
	defer wc.mu.Unlock()

	mem, ok := <-wc.memMu
	if !ok {
		return nil, minimizeResponse{}, errSharedMemClosed
	}
	mem.header().count = 0
	mem.setValue(valueIn)
	wc.memMu <- mem

	c := call{Minimize: &args}
	err = wc.call(ctx, c, &resp)
	mem, ok = <-wc.memMu
	if !ok {
		return nil, minimizeResponse{}, errSharedMemClosed
	}
	valueOut = mem.valueCopy()
	resp.Count = mem.header().count
	wc.memMu <- mem

	return valueOut, resp, err
}

// fuzz tells the worker to call the fuzz method. See workerServer.fuzz.
func (wc *workerClient) fuzz(ctx context.Context, valueIn []byte, args fuzzArgs) (valueOut []byte, resp fuzzResponse, err error) {
	wc.mu.Lock()
	defer wc.mu.Unlock()

	mem, ok := <-wc.memMu
	if !ok {
		return nil, fuzzResponse{}, errSharedMemClosed
	}
	mem.header().count = 0
	mem.setValue(valueIn)
	wc.memMu <- mem

	c := call{Fuzz: &args}
	err = wc.call(ctx, c, &resp)
	mem, ok = <-wc.memMu
	if !ok {
		return nil, fuzzResponse{}, errSharedMemClosed
	}
	valueOut = mem.valueCopy()
	resp.Count = mem.header().count
	wc.memMu <- mem

	return valueOut, resp, err
}

// ping tells the worker to call the ping method. See workerServer.ping.
func (wc *workerClient) ping(ctx context.Context) error {
	c := call{Ping: &pingArgs{}}
	var resp pingResponse
	return wc.call(ctx, c, &resp)
}

// call sends an RPC from the coordinator to the worker process and waits for
// the response. The call may be cancelled with ctx.
func (wc *workerClient) call(ctx context.Context, c call, resp interface{}) (err error) {
	// This goroutine may stay blocked after call returns because the underlying
	// read blocks, even after the file descriptor in this process is closed. The
	// pipe must be closed by the server, too.
	errC := make(chan error, 1)
	go func() {
		if err := wc.enc.Encode(c); err != nil {
			errC <- err
			return
		}
		errC <- wc.dec.Decode(resp)
	}()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-errC:
		return err
	}
}
