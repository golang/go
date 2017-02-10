// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"runtime"
	"sync"
	"syscall"
)

// asyncIO implements asynchronous cancelable I/O.
// An asyncIO represents a single asynchronous Read or Write
// operation. The result is returned on the result channel.
// The undergoing I/O system call can either complete or be
// interrupted by a note.
type asyncIO struct {
	res chan result

	// mu guards the pid field.
	mu sync.Mutex

	// pid holds the process id of
	// the process running the IO operation.
	pid int
}

// result is the return value of a Read or Write operation.
type result struct {
	n   int
	err error
}

// newAsyncIO returns a new asyncIO that performs an I/O
// operation by calling fn, which must do one and only one
// interruptible system call.
func newAsyncIO(fn func([]byte) (int, error), b []byte) *asyncIO {
	aio := &asyncIO{
		res: make(chan result, 0),
	}
	aio.mu.Lock()
	go func() {
		// Lock the current goroutine to its process
		// and store the pid in io so that Cancel can
		// interrupt it. We ignore the "hangup" signal,
		// so the signal does not take down the entire
		// Go runtime.
		runtime.LockOSThread()
		runtime_ignoreHangup()
		aio.pid = syscall.Getpid()
		aio.mu.Unlock()

		n, err := fn(b)

		aio.mu.Lock()
		aio.pid = -1
		runtime_unignoreHangup()
		aio.mu.Unlock()

		aio.res <- result{n, err}
	}()
	return aio
}

// Cancel interrupts the I/O operation, causing
// the Wait function to return.
func (aio *asyncIO) Cancel() {
	aio.mu.Lock()
	defer aio.mu.Unlock()
	if aio.pid == -1 {
		return
	}
	f, e := syscall.Open("/proc/"+itoa(aio.pid)+"/note", syscall.O_WRONLY)
	if e != nil {
		return
	}
	syscall.Write(f, []byte("hangup"))
	syscall.Close(f)
}

// Wait for the I/O operation to complete.
func (aio *asyncIO) Wait() (int, error) {
	res := <-aio.res
	return res.n, res.err
}

// The following functions, provided by the runtime, are used to
// ignore and unignore the "hangup" signal received by the process.
func runtime_ignoreHangup()
func runtime_unignoreHangup()
