// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"internal/synctest"
	"internal/syscall/windows"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
)

// ioCancelState coordinates cancellation with synchronous I/O workers.
// Its zero value is ready for use. The FD's directional locks permit at most
// one read worker and one write worker at a time.
type ioCancelState struct {
	canceling  atomic.Bool
	cancelDone sync.WaitGroup
	// A worker clears its slot once its I/O returns, before sending the result.
	workers [2]atomic.Pointer[syncIOWorkerState]
}

// begin publishes the worker before its closing check and returns its slot.
// The worker must call end after I/O returns and before sending the result.
func (s *ioCancelState) begin(worker *syncIOWorkerState) int {
	for i := range s.workers {
		if s.workers[i].CompareAndSwap(nil, worker) {
			return i
		}
	}
	// A caller bypassed a directional lock, or a worker did not clear its slot.
	panic("too many synchronous I/O workers")
}

func (s *ioCancelState) end(slot int) {
	// Clear the slot before waiting, so cancellation retries can finish.
	s.workers[slot].Store(nil)
	if s.canceling.Load() {
		s.cancelDone.Wait()
	}
}

// cancelIO requests cancellation of outstanding I/O. Close waits for completion
// by waiting for the I/O callers to release their FD references.
// The caller must have marked the FD closed and hold a reference to it.
// Only one goroutine may call cancelIO.
func (fd *FD) cancelIO() {
	s := &fd.ioCancel
	// Workers clear their slots before waiting in end, so retries can finish
	// without waiting for result delivery. Add before publishing canceling, so
	// a worker seen by the scan cannot return from end before cancellation ends.
	s.cancelDone.Add(1)
	s.canceling.Store(true)
	defer s.cancelDone.Done()

	// Workers published after a scan see closing and skip I/O. We only
	// need to remember successful cancellation for each slot.
	var canceled [2]bool
	first := true
	for {
		var retry bool
		for i := range s.workers {
			worker := s.workers[i].Load()
			if worker == nil || canceled[i] {
				continue
			}
			switch err := windows.CancelSynchronousIo(worker.thread); err {
			case nil:
				// The thread has only this I/O request to cancel. end keeps
				// it alive and unavailable for reuse until cancelIO returns.
				canceled[i] = true
			case syscall.ERROR_NOT_FOUND:
				// I/O may have returned, or may not have been submitted yet.
				retry = true
			default:
				panic(err)
			}
		}
		if first {
			first = false
			// Also cancel overlapped and externally issued I/O, including I/O
			// blocking lazy initialization, even if no worker is published.
			// Target workers first so their cancellation is known separately.
			if err := syscall.CancelIoEx(fd.Sysfd, nil); err != nil && err != syscall.ERROR_NOT_FOUND {
				panic(err)
			}
		}
		if !retry {
			return
		}
		// ERROR_NOT_FOUND may mean a worker passed its closing check but
		// has not submitted I/O yet. It can still submit a blocking request,
		// so we must retry. Alternatively, I/O has completed but the worker
		// has not cleared its slot yet. Yield to let the worker make progress.
		runtime.Gosched()
	}
}

// runtime_startThread starts a goroutine locked to a fresh OS thread. The
// thread exits when the function returns; it must not call UnlockOSThread.
func runtime_startThread(func())

// runtime_threadHandle returns the runtime-owned handle of the current worker
// thread. It may only be called from a runtime_startThread worker. The handle
// is valid until the worker returns and must not be closed by the caller.
func runtime_threadHandle() syscall.Handle

type syncIORequest struct {
	fd     *FD
	submit func(syscall.Handle, []byte, *uint32, *syscall.Overlapped) error
	buf    []byte
}

type syncIOResult struct {
	qty uint32
	err error
}

type syncIOWorker struct {
	state *syncIOWorkerState
}

// The running worker references only its state, not the cached owner. This
// lets the owner's cleanup stop the thread when sync.Pool discards it.
type syncIOWorkerState struct {
	requests chan syncIORequest
	results  chan syncIOResult
	thread   syscall.Handle // borrowed from runtime; set before begin
}

// Reuse idle workers through sync.Pool's per-P caches. Active workers are not
// limited: they may all be blocked in Read while another must run Write.
// Discarded owners shut down their idle workers through a cleanup.
var idleSyncIOWorkers sync.Pool

func acquireSyncIOWorker() *syncIOWorker {
	// A worker created in a synctest bubble must exit within that bubble;
	// neither its goroutine nor its channels may enter the shared cache.
	inBubble := synctest.IsInBubble()
	if !inBubble {
		if worker := idleSyncIOWorkers.Get(); worker != nil {
			return worker.(*syncIOWorker)
		}
	}
	state := &syncIOWorkerState{
		requests: make(chan syncIORequest),
		results:  make(chan syncIOResult),
	}
	runtime_startThread(state.run)
	worker := &syncIOWorker{state: state}
	if !inBubble {
		runtime.AddCleanup(worker, (*syncIOWorkerState).stop, state)
	}
	return worker
}

func (worker *syncIOWorker) release() {
	if synctest.IsInBubble() {
		worker.state.stop()
	} else {
		idleSyncIOWorkers.Put(worker)
	}
	runtime.KeepAlive(worker)
}

func (worker *syncIOWorkerState) stop() {
	close(worker.requests)
}

func (worker *syncIOWorkerState) run() {
	worker.thread = runtime_threadHandle()

	// Reuse the result because passing its address to submit makes it escape.
	var result syncIOResult
	for request := range worker.requests {
		slot := request.fd.ioCancel.begin(worker)
		if request.fd.closing() {
			result.err = errClosing(request.fd.isFile)
		} else {
			result.err = request.submit(request.fd.Sysfd, request.buf, &result.qty, nil)
		}
		// Wait for cancellation before the result lets the caller reuse or
		// stop this worker.
		request.fd.ioCancel.end(slot)
		// Do not retain the caller's file or buffer while this worker is idle.
		request = syncIORequest{}
		worker.results <- result
		result = syncIOResult{}
	}
}

// execSyncIO queues synchronous pipe I/O to a dedicated worker. The caller
// retains the I/O lock, FD reference, and buffer until the worker returns,
// but does not need to reserve its own OS thread.
func (fd *FD) execSyncIO(submit func(syscall.Handle, []byte, *uint32, *syscall.Overlapped) error, buf []byte) (uint32, error) {
	worker := acquireSyncIOWorker()
	worker.state.requests <- syncIORequest{fd, submit, buf}
	result := <-worker.state.results
	worker.release()
	return result.qty, result.err
}
