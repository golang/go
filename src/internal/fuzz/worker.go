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
// coordinate loops until ctx is cancelled or a fatal error is encountered. While
// looping, coordinate receives inputs from w.coordinator.inputC, then passes
// those on to the worker process.
func (w *worker) coordinate(ctx context.Context) error {
	// Start the process.
	if err := w.start(); err != nil {
		// We couldn't start the worker process. We can't do anything, and it's
		// likely that other workers can't either, so don't try to restart.
		return err
	}

	// Send the worker a message to make sure it can respond.
	// Errors that occur before we get a response likely indicate that
	// the worker did not call F.Fuzz or called F.Fail first.
	// We don't record crashers for these errors.
	if err := w.client.ping(ctx); err != nil {
		w.stop()
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if isInterruptError(err) {
			// User may have pressed ^C before worker responded.
			return nil
		}
		return fmt.Errorf("fuzzing process terminated without fuzzing: %w", err)
		// TODO(jayconrod,katiehockman): record and return stderr.
	}

	// interestingCount starts at -1, like the coordinator does, so that the
	// worker client's coverage data is updated after a coverage-only run.
	interestingCount := int64(-1)

	// Main event loop.
	for {
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
			// TODO(jayconrod,katiehockman): record and return stderr.

		case input := <-w.coordinator.inputC:
			// Received input from coordinator.
			args := fuzzArgs{Count: input.countRequested, Duration: workerFuzzDuration, CoverageOnly: input.coverageOnly}
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
				// Unexpected termination. Attempt to minimize, then inform the
				// coordinator about the crash.
				// TODO(jayconrod,katiehockman): if -keepfuzzing, restart worker.
				// TODO(jayconrod,katiehockman): consider informing the
				// coordinator that this worker is minimizing, in order to block
				// the other workers from receiving more inputs.
				message := fmt.Sprintf("fuzzing process terminated unexpectedly: %v", w.waitErr)
				err = w.waitErr
				res, minimized, minErr := w.minimize(ctx)
				if !minimized {
					// Minimization did not find a smaller crashing value, so
					// return the one we already found.
					res = fuzzResult{
						entry:      CorpusEntry{Data: value},
						crasherMsg: message,
					}
				}
				if minErr != nil {
					err = minErr
				}
				w.coordinator.resultC <- res
				return err
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
		}
	}
}

// minimize asks a workerServer to attempt to minimize what is currently in
// shared memory. It runs for a maxium of 1 minute. The worker must be stopped
// when minimize is called.
func (w *worker) minimize(ctx context.Context) (res fuzzResult, minimized bool, retErr error) {
	fmt.Fprint(w.coordinator.opts.Log, "found a crash, currently minimizing for up to 1 minute\n")
	defer func() {
		w.stop()
		if retErr == nil {
			retErr = w.waitErr
		}
	}()
	// In case we can't minimize it at all, save the last crash value that we
	// found to send to the coordinator once the time is up.
	minimizeDeadline := time.Now().Add(time.Minute)
	for rem := time.Until(minimizeDeadline); rem > 0; {
		// Restart the worker.
		if err := w.start(); err != nil {
			return res, minimized, err
		}
		args := minimizeArgs{Duration: rem}
		value, err := w.client.minimize(ctx, args)
		if err == nil {
			// Minimization finished successfully, meaning that it
			// couldn't find any smaller inputs that caused a crash,
			// so stop trying.
			return res, minimized, nil
		}
		// Minimization will return an error for a non-recoverable problem, so
		// a non-nil error is expected. However, make sure it didn't fail for
		// some other reason which should cause us to stop minimizing.
		if ctx.Err() != nil || w.interrupted || isInterruptError(w.waitErr) {
			return res, minimized, nil
		}

		// The bytes in memory caused a legitimate crash, so stop the worker and
		// save this value and error message.
		w.stop()
		message := fmt.Sprintf("fuzzing process terminated unexpectedly: %v", w.waitErr)
		res = fuzzResult{
			entry:      CorpusEntry{Data: value},
			crasherMsg: message,
		}
		minimized = true
	}
	return res, minimized, nil
	// TODO(jayconrod,katiehockman): while minimizing, every panic message is
	// logged to STDOUT. We should probably suppress all but the last one to
	// lower the noise.
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
	if w.cmd != nil {
		panic("worker already started")
	}
	w.waitErr = nil
	w.interrupted = false
	w.termC = nil

	cmd := exec.Command(w.binPath, w.args...)
	cmd.Dir = w.dir
	cmd.Env = w.env[:len(w.env):len(w.env)] // copy on append to ensure workers don't overwrite each other.
	// TODO(jayconrod): set stdout and stderr to nil or buffer. A large number
	// of workers may be very noisy, but for now, this output is useful for
	// debugging.
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// TODO(jayconrod): set up shared memory between the coordinator and worker to
	// transfer values and coverage data. If the worker crashes, we need to be
	// able to find the value that caused the crash.

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
	Duration time.Duration
}

// minimizeResponse contains results from workerServer.minimize.
type minimizeResponse struct{}

// fuzzArgs contains arguments to workerServer.fuzz. The value to fuzz is
// passed in shared memory.
type fuzzArgs struct {
	// Duration is the time to spend fuzzing, not including starting or
	// cleaning up.
	Duration time.Duration

	// Count is the number of values to test, without spending more time
	// than Duration.
	Count int64

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

	fuzzCtx, cancel := context.WithTimeout(ctx, args.Duration)
	defer cancel()
	mem := <-ws.memMu
	defer func() { ws.memMu <- mem }()

	vals, err := unmarshalCorpusFile(mem.valueCopy())
	if err != nil {
		panic(err)
	}

	if args.CoverageOnly {
		// Reset the coverage each time before running the fuzzFn.
		resetCoverage()
		ws.fuzzFn(CorpusEntry{Values: vals})
		resp.CoverageData = coverageCopy()
		return resp
	}

	cov := coverage()
	if len(cov) != len(ws.coverageData) {
		panic(fmt.Sprintf("num edges changed at runtime: %d, expected %d", len(cov), len(ws.coverageData)))
	}
	for {
		select {
		case <-fuzzCtx.Done():
			return resp

		default:
			resp.Count++
			ws.m.mutate(vals, cap(mem.valueRef()))
			writeToMem(vals, mem)
			resetCoverage()
			if err := ws.fuzzFn(CorpusEntry{Values: vals}); err != nil {
				// TODO(jayconrod,katiehockman): consider making the maximum
				// minimization time customizable with a go command flag.
				minCtx, minCancel := context.WithTimeout(ctx, time.Minute)
				defer minCancel()
				if minErr := ws.minimizeInput(minCtx, vals, mem); minErr != nil {
					// Minimization found a different error, so use that one.
					err = minErr
				}
				resp.Err = err.Error()
				if resp.Err == "" {
					resp.Err = "fuzz function failed with no output"
				}
				return resp
			}
			for i := range cov {
				if ws.coverageData[i] == 0 && cov[i] > ws.coverageData[i] {
					// TODO(jayconrod,katie): minimize this.
					// This run hit a new edge. Only allocate a new slice as a
					// copy of cov if we are returning, since it is expensive.
					resp.CoverageData = coverageCopy()
					return resp
				}
			}
			if args.Count > 0 && resp.Count == args.Count {
				return resp
			}
		}
	}
}

func (ws *workerServer) minimize(ctx context.Context, args minimizeArgs) minimizeResponse {
	mem := <-ws.memMu
	defer func() { ws.memMu <- mem }()
	vals, err := unmarshalCorpusFile(mem.valueCopy())
	if err != nil {
		panic(err)
	}
	ctx, cancel := context.WithTimeout(ctx, args.Duration)
	defer cancel()
	ws.minimizeInput(ctx, vals, mem)
	return minimizeResponse{}
}

// minimizeInput applies a series of minimizing transformations on the provided
// vals, ensuring that each minimization still causes an error in fuzzFn. Before
// every call to fuzzFn, it marshals the new vals and writes it to the provided
// mem just in case an unrecoverable error occurs. It uses the context to
// determine how long to run, stopping once closed. It returns the last error it
// found.
func (ws *workerServer) minimizeInput(ctx context.Context, vals []interface{}, mem *sharedMem) (retErr error) {
	// Make sure the last crashing value is written to mem.
	defer writeToMem(vals, mem)

	// tryMinimized will run the fuzz function for the values in vals at the
	// time the function is called. If err is nil, then the minimization was
	// unsuccessful, since we expect an error to still occur.
	tryMinimized := func(i int, prevVal interface{}) error {
		writeToMem(vals, mem) // write to mem in case a non-recoverable crash occurs
		err := ws.fuzzFn(CorpusEntry{Values: vals})
		if err == nil {
			// The fuzz function succeeded, so return the value at index i back
			// to the previously failing input.
			vals[i] = prevVal
		} else {
			// The fuzz function failed, so save the most recent error.
			retErr = err
		}
		return err
	}
	for valI := range vals {
		switch v := vals[valI].(type) {
		case bool, byte, rune:
			continue // can't minimize
		case string, int, int8, int16, int64, uint, uint16, uint32, uint64, float32, float64:
			// TODO(jayconrod,katiehockman): support minimizing other types
		case []byte:
			// First, try to cut the tail.
			for n := 1024; n != 0; n /= 2 {
				for len(v) > n {
					if ctx.Err() != nil {
						return retErr
					}
					vals[valI] = v[:len(v)-n]
					if tryMinimized(valI, v) == nil {
						break
					}
					// Set v to the new value to continue iterating.
					v = v[:len(v)-n]
				}
			}

			// Then, try to remove each individual byte.
			tmp := make([]byte, len(v))
			for i := 0; i < len(v)-1; i++ {
				if ctx.Err() != nil {
					return retErr
				}
				candidate := tmp[:len(v)-1]
				copy(candidate[:i], v[:i])
				copy(candidate[i:], v[i+1:])
				vals[valI] = candidate
				if tryMinimized(valI, v) == nil {
					continue
				}
				// Update v to delete the value at index i.
				copy(v[i:], v[i+1:])
				v = v[:len(candidate)]
				// v[i] is now different, so decrement i to redo this iteration
				// of the loop with the new value.
				i--
			}

			// Then, try to remove each possible subset of bytes.
			for i := 0; i < len(v)-1; i++ {
				copy(tmp, v[:i])
				for j := len(v); j > i+1; j-- {
					if ctx.Err() != nil {
						return retErr
					}
					candidate := tmp[:len(v)-j+i]
					copy(candidate[i:], v[j:])
					vals[valI] = candidate
					if tryMinimized(valI, v) == nil {
						continue
					}
					// Update v and reset the loop with the new length.
					copy(v[i:], v[j:])
					v = v[:len(candidate)]
					j = len(v)
				}
			}
			// TODO(jayconrod,katiehockman): consider adding canonicalization
			// which replaces each individual byte with '0'
		default:
			panic("unreachable")
		}
	}
	return retErr
}

func writeToMem(vals []interface{}, mem *sharedMem) {
	b := marshalCorpusFile(vals...)
	mem.setValueLen(len(b))
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
func (wc *workerClient) minimize(ctx context.Context, args minimizeArgs) (valueOut []byte, err error) {
	wc.mu.Lock()
	defer wc.mu.Unlock()

	var resp minimizeResponse
	c := call{Minimize: &args}
	err = wc.call(ctx, c, &resp)
	mem, ok := <-wc.memMu
	if !ok {
		return nil, errSharedMemClosed
	}
	valueOut = mem.valueCopy()
	wc.memMu <- mem

	return valueOut, err
}

// fuzz tells the worker to call the fuzz method. See workerServer.fuzz.
func (wc *workerClient) fuzz(ctx context.Context, valueIn []byte, args fuzzArgs) (valueOut []byte, resp fuzzResponse, err error) {
	wc.mu.Lock()
	defer wc.mu.Unlock()

	mem, ok := <-wc.memMu
	if !ok {
		return nil, fuzzResponse{}, errSharedMemClosed
	}
	mem.setValue(valueIn)
	wc.memMu <- mem

	c := call{Fuzz: &args}
	err = wc.call(ctx, c, &resp)
	mem, ok = <-wc.memMu
	if !ok {
		return nil, fuzzResponse{}, errSharedMemClosed
	}
	valueOut = mem.valueCopy()
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
