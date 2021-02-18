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

	cmd     *exec.Cmd     // current worker process
	client  *workerClient // used to communicate with worker process
	waitErr error         // last error returned by wait, set before termC is closed.
	termC   chan struct{} // closed by wait when worker process terminates
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

// runFuzzing runs the test binary to perform fuzzing.
//
// This function loops until w.coordinator.doneC is closed or some
// fatal error is encountered. It receives inputs from w.coordinator.inputC,
// then passes those on to the worker process.
func (w *worker) runFuzzing() error {
	// Start the process.
	if err := w.start(); err != nil {
		// We couldn't start the worker process. We can't do anything, and it's
		// likely that other workers can't either, so give up.
		w.coordinator.errC <- err
		return err
	}

	inputC := w.coordinator.inputC // set to nil when processing input
	fuzzC := make(chan struct{})   // sent when we finish processing an input.

	// Main event loop.
	for {
		select {
		case <-w.coordinator.doneC:
			// All workers were told to stop.
			err := w.stop()
			if isInterruptError(err) {
				// Worker interrupted by SIGINT. This can happen if the worker receives
				// SIGINT before installing the signal handler. That's likely if
				// TestMain or the fuzz target setup takes a long time.
				return nil
			}
			return err

		case <-w.termC:
			// Worker process terminated unexpectedly.
			if isInterruptError(w.waitErr) {
				// Worker interrupted by SIGINT. See comment in doneC case.
				w.stop()
				return nil
			}
			if w.waitErr == nil {
				// Worker exited 0.
				w.stop()
				return fmt.Errorf("worker exited unexpectedly with status 0")
			}

			// Unexpected termination. Inform the coordinator about the crash.
			// TODO(jayconrod,katiehockman): if -keepfuzzing, restart worker.
			mem := <-w.memMu
			value := mem.valueCopy()
			w.memMu <- mem
			message := fmt.Sprintf("fuzzing process terminated unexpectedly: %v", w.waitErr)
			crasher := crasherEntry{
				CorpusEntry: CorpusEntry{Data: value},
				errMsg:      message,
			}
			w.coordinator.crasherC <- crasher
			return w.stop()

		case input := <-inputC:
			// Received input from coordinator.
			inputC = nil // block new inputs until we finish with this one.
			go func() {
				args := fuzzArgs{Duration: workerFuzzDuration}
				value, resp, err := w.client.fuzz(input.Data, args)
				if err != nil {
					// Error communicating with worker.
					select {
					case <-w.termC:
						// Worker terminated, perhaps unexpectedly.
						// We expect I/O errors due to partially sent or received RPCs,
						// so ignore this error.
					case <-w.coordinator.doneC:
						// Timeout or interruption. Worker may also be interrupted.
						// Again, ignore I/O errors.
					default:
						// TODO(jayconrod): if we get an error here, something failed between
						// main and the call to testing.F.Fuzz. The error here won't
						// be useful. Collect stderr, clean it up, and return that.
						// TODO(jayconrod): we can get EPIPE if w.stop is called concurrently
						// and it kills the worker process. Suppress this message in
						// that case.
						fmt.Fprintf(os.Stderr, "communicating with worker: %v\n", err)
					}
					// TODO(jayconrod): what happens if testing.F.Fuzz is never called?
					// TODO(jayconrod): time out if the test process hangs.
				} else if resp.Err != "" {
					// The worker found a crasher. Inform the coordinator.
					crasher := crasherEntry{
						CorpusEntry: CorpusEntry{Data: value},
						errMsg:      resp.Err,
					}
					w.coordinator.crasherC <- crasher
				} else {
					// Inform the coordinator that fuzzing found something
					// interesting (i.e. new coverage).
					if resp.Interesting {
						w.coordinator.interestingC <- CorpusEntry{Data: value}
					}

					// Continue fuzzing.
					fuzzC <- struct{}{}
				}
				// TODO(jayconrod,katiehockman): gather statistics.
			}()

		case <-fuzzC:
			// Worker finished fuzzing and nothing new happened.
			inputC = w.coordinator.inputC // unblock new inputs
		}
	}
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
	w.termC = nil

	cmd := exec.Command(w.binPath, w.args...)
	cmd.Dir = w.dir
	cmd.Env = w.env
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
// stop must be called once after start returns successfully, even if the
// worker process terminates unexpectedly.
func (w *worker) stop() error {
	if w.termC == nil {
		panic("worker was not started successfully")
	}
	select {
	case <-w.termC:
		// Worker already terminated, perhaps unexpectedly.
		if w.client == nil {
			panic("worker already stopped")
		}
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
				fmt.Fprintf(os.Stderr, "go: waiting for fuzz worker to terminate...\n")
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
	Fuzz *fuzzArgs
}

// fuzzArgs contains arguments to workerServer.fuzz. The value to fuzz is
// passed in shared memory.
type fuzzArgs struct {
	Duration time.Duration
}

// fuzzResponse contains results from workerServer.fuzz.
type fuzzResponse struct {
	// Interesting indicates the value in shared memory may be interesting to
	// the coordinator (for example, because it expanded coverage).
	Interesting bool

	// Err is set if the value in shared memory caused a crash.
	Err string
}

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
	// Stop handling messages when ctx.Done() is closed. This normally happens
	// when the worker process receives a SIGINT signal, which on POSIX platforms
	// is sent to the process group when ^C is pressed.
	//
	// Ordinarily, the coordinator process may stop a worker by closing fuzz_in.
	// We simulate that and interrupt a blocked read here.
	doneC := make(chan struct{})
	defer func() { close(doneC) }()
	go func() {
		select {
		case <-ctx.Done():
			ws.fuzzIn.Close()
		case <-doneC:
		}
	}()

	enc := json.NewEncoder(ws.fuzzOut)
	dec := json.NewDecoder(ws.fuzzIn)
	for {
		var c call
		if err := dec.Decode(&c); err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			} else if err == io.EOF {
				return nil
			} else {
				return err
			}
		}

		var resp interface{}
		switch {
		case c.Fuzz != nil:
			resp = ws.fuzz(ctx, *c.Fuzz)
		default:
			return errors.New("no arguments provided for any call")
		}

		if err := enc.Encode(resp); err != nil {
			return err
		}
	}
}

// fuzz runs the test function on random variations of a given input value for
// a given amount of time. fuzz returns early if it finds an input that crashes
// the fuzz function or an input that expands coverage.
func (ws *workerServer) fuzz(ctx context.Context, args fuzzArgs) fuzzResponse {
	ctx, cancel := context.WithTimeout(ctx, args.Duration)
	defer cancel()
	mem := <-ws.memMu
	defer func() { ws.memMu <- mem }()

	vals, err := unmarshalCorpusFile(mem.valueCopy())
	if err != nil {
		panic(err)
	}
	for {
		select {
		case <-ctx.Done():
			// TODO(jayconrod,katiehockman): this value is not interesting. Use a
			// real heuristic once we have one.
			return fuzzResponse{Interesting: true}
		default:
			vals = ws.m.mutate(vals, cap(mem.valueRef()))
			b := marshalCorpusFile(vals...)
			mem.setValueLen(len(b))
			mem.setValue(b)
			if err := ws.fuzzFn(CorpusEntry{Values: vals}); err != nil {
				return fuzzResponse{Err: err.Error()}
			}
			// TODO(jayconrod,katiehockman): return early if we find an
			// interesting value.
		}
	}
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

// fuzz tells the worker to call the fuzz method. See workerServer.fuzz.
func (wc *workerClient) fuzz(valueIn []byte, args fuzzArgs) (valueOut []byte, resp fuzzResponse, err error) {
	wc.mu.Lock()
	defer wc.mu.Unlock()

	mem, ok := <-wc.memMu
	if !ok {
		return nil, fuzzResponse{}, errSharedMemClosed
	}
	mem.setValue(valueIn)
	wc.memMu <- mem

	c := call{Fuzz: &args}
	if err := wc.enc.Encode(c); err != nil {
		return nil, fuzzResponse{}, err
	}
	err = wc.dec.Decode(&resp)

	mem, ok = <-wc.memMu
	if !ok {
		return nil, fuzzResponse{}, errSharedMemClosed
	}
	valueOut = mem.valueCopy()
	wc.memMu <- mem

	return valueOut, resp, err
}
