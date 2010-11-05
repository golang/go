// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Pipe adapter to connect code expecting an io.Reader
// with code expecting an io.Writer.

package io

import (
	"os"
	"runtime"
	"sync"
)

type pipeResult struct {
	n   int
	err os.Error
}

// Shared pipe structure.
type pipe struct {
	// Reader sends on cr1, receives on cr2.
	// Writer does the same on cw1, cw2.
	r1, w1 chan []byte
	r2, w2 chan pipeResult

	rclose chan os.Error // read close; error to return to writers
	wclose chan os.Error // write close; error to return to readers

	done chan int // read or write half is done
}

func (p *pipe) run() {
	var (
		rb    []byte      // pending Read
		wb    []byte      // pending Write
		wn    int         // amount written so far from wb
		rerr  os.Error    // if read end is closed, error to send to writers
		werr  os.Error    // if write end is closed, error to send to readers
		r1    chan []byte // p.cr1 or nil depending on whether Read is ok
		w1    chan []byte // p.cw1 or nil depending on whether Write is ok
		ndone int
	)

	// Read and Write are enabled at the start.
	r1 = p.r1
	w1 = p.w1

	for {
		select {
		case <-p.done:
			if ndone++; ndone == 2 {
				// both reader and writer are gone
				// close out any existing i/o
				if r1 == nil {
					p.r2 <- pipeResult{0, os.EINVAL}
				}
				if w1 == nil {
					p.w2 <- pipeResult{0, os.EINVAL}
				}
				return
			}
			continue
		case rerr = <-p.rclose:
			if w1 == nil {
				// finish pending Write
				p.w2 <- pipeResult{wn, rerr}
				wn = 0
				w1 = p.w1 // allow another Write
			}
			if r1 == nil {
				// Close of read side during Read.
				// finish pending Read with os.EINVAL.
				p.r2 <- pipeResult{0, os.EINVAL}
				r1 = p.r1 // allow another Read
			}
			continue
		case werr = <-p.wclose:
			if r1 == nil {
				// finish pending Read
				p.r2 <- pipeResult{0, werr}
				r1 = p.r1 // allow another Read
			}
			if w1 == nil {
				// Close of write side during Write.
				// finish pending Write with os.EINVAL.
				p.w2 <- pipeResult{wn, os.EINVAL}
				wn = 0
				w1 = p.w1 // allow another Write
			}
			continue
		case rb = <-r1:
			if werr != nil {
				// write end is closed
				p.r2 <- pipeResult{0, werr}
				continue
			}
			if rerr != nil {
				// read end is closed
				p.r2 <- pipeResult{0, os.EINVAL}
				continue
			}
			r1 = nil // disable Read until this one is done
		case wb = <-w1:
			if rerr != nil {
				// read end is closed
				p.w2 <- pipeResult{0, rerr}
				continue
			}
			if werr != nil {
				// write end is closed
				p.w2 <- pipeResult{0, os.EINVAL}
				continue
			}
			w1 = nil // disable Write until this one is done
		}

		if r1 == nil && w1 == nil {
			// Have rb and wb.  Execute.
			n := copy(rb, wb)
			wn += n
			wb = wb[n:]

			// Finish Read.
			p.r2 <- pipeResult{n, nil}
			r1 = p.r1 // allow another Read

			// Maybe finish Write.
			if len(wb) == 0 {
				p.w2 <- pipeResult{wn, nil}
				wn = 0
				w1 = p.w1 // allow another Write
			}
		}
	}
}

// Read/write halves of the pipe.
// They are separate structures for two reasons:
//  1.  If one end becomes garbage without being Closed,
//      its finalizer can Close so that the other end
//      does not hang indefinitely.
//  2.  Clients cannot use interface conversions on the
//      read end to find the Write method, and vice versa.

type pipeHalf struct {
	c1     chan []byte
	c2     chan pipeResult
	cclose chan os.Error
	done   chan int

	lock   sync.Mutex
	closed bool

	io       sync.Mutex
	ioclosed bool
}

func (p *pipeHalf) rw(data []byte) (n int, err os.Error) {
	// Run i/o operation.
	// Check ioclosed flag under lock to make sure we're still allowed to do i/o.
	p.io.Lock()
	if p.ioclosed {
		p.io.Unlock()
		return 0, os.EINVAL
	}
	p.io.Unlock()
	p.c1 <- data
	res := <-p.c2
	return res.n, res.err
}

func (p *pipeHalf) close(err os.Error) os.Error {
	// Close pipe half.
	// Only first call to close does anything.
	p.lock.Lock()
	if p.closed {
		p.lock.Unlock()
		return os.EINVAL
	}
	p.closed = true
	p.lock.Unlock()

	// First, send the close notification.
	p.cclose <- err

	// Runner is now responding to rw operations
	// with os.EINVAL.  Cut off future rw operations
	// by setting ioclosed flag.
	p.io.Lock()
	p.ioclosed = true
	p.io.Unlock()

	// With ioclosed set, there will be no more rw operations
	// working on the channels.
	// Tell the runner we won't be bothering it anymore.
	p.done <- 1

	// Successfully torn down; can disable finalizer.
	runtime.SetFinalizer(p, nil)

	return nil
}

func (p *pipeHalf) finalizer() {
	p.close(os.EINVAL)
}


// A PipeReader is the read half of a pipe.
type PipeReader struct {
	pipeHalf
}

// Read implements the standard Read interface:
// it reads data from the pipe, blocking until a writer
// arrives or the write end is closed.
// If the write end is closed with an error, that error is
// returned as err; otherwise err is nil.
func (r *PipeReader) Read(data []byte) (n int, err os.Error) {
	return r.rw(data)
}

// Close closes the reader; subsequent writes to the
// write half of the pipe will return the error os.EPIPE.
func (r *PipeReader) Close() os.Error {
	return r.CloseWithError(nil)
}

// CloseWithError closes the reader; subsequent writes
// to the write half of the pipe will return the error err.
func (r *PipeReader) CloseWithError(err os.Error) os.Error {
	if err == nil {
		err = os.EPIPE
	}
	return r.close(err)
}

// A PipeWriter is the write half of a pipe.
type PipeWriter struct {
	pipeHalf
}

// Write implements the standard Write interface:
// it writes data to the pipe, blocking until readers
// have consumed all the data or the read end is closed.
// If the read end is closed with an error, that err is
// returned as err; otherwise err is os.EPIPE.
func (w *PipeWriter) Write(data []byte) (n int, err os.Error) {
	return w.rw(data)
}

// Close closes the writer; subsequent reads from the
// read half of the pipe will return no bytes and os.EOF.
func (w *PipeWriter) Close() os.Error {
	return w.CloseWithError(nil)
}

// CloseWithError closes the writer; subsequent reads from the
// read half of the pipe will return no bytes and the error err.
func (w *PipeWriter) CloseWithError(err os.Error) os.Error {
	if err == nil {
		err = os.EOF
	}
	return w.close(err)
}

// Pipe creates a synchronous in-memory pipe.
// It can be used to connect code expecting an io.Reader
// with code expecting an io.Writer.
// Reads on one end are matched with writes on the other,
// copying data directly between the two; there is no internal buffering.
func Pipe() (*PipeReader, *PipeWriter) {
	p := &pipe{
		r1:     make(chan []byte),
		r2:     make(chan pipeResult),
		w1:     make(chan []byte),
		w2:     make(chan pipeResult),
		rclose: make(chan os.Error),
		wclose: make(chan os.Error),
		done:   make(chan int),
	}
	go p.run()

	// NOTE: Cannot use composite literal here:
	//	pipeHalf{c1: p.cr1, c2: p.cr2, cclose: p.crclose, cdone: p.cdone}
	// because this implicitly copies the pipeHalf, which copies the inner mutex.

	r := new(PipeReader)
	r.c1 = p.r1
	r.c2 = p.r2
	r.cclose = p.rclose
	r.done = p.done
	runtime.SetFinalizer(r, (*PipeReader).finalizer)

	w := new(PipeWriter)
	w.c1 = p.w1
	w.c2 = p.w2
	w.cclose = p.wclose
	w.done = p.done
	runtime.SetFinalizer(w, (*PipeWriter).finalizer)

	return r, w
}
