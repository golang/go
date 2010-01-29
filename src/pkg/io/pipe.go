// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Pipe adapter to connect code expecting an io.Read
// with code expecting an io.Write.

package io

import (
	"os"
	"sync"
)

type pipeReturn struct {
	n   int
	err os.Error
}

// Shared pipe structure.
type pipe struct {
	rclosed bool            // Read end closed?
	rerr    os.Error        // Error supplied to CloseReader
	wclosed bool            // Write end closed?
	werr    os.Error        // Error supplied to CloseWriter
	wpend   []byte          // Written data waiting to be read.
	wtot    int             // Bytes consumed so far in current write.
	cr      chan []byte     // Write sends data here...
	cw      chan pipeReturn // ... and reads the n, err back from here.
}

func (p *pipe) Read(data []byte) (n int, err os.Error) {
	if p == nil || p.rclosed {
		return 0, os.EINVAL
	}

	// Wait for next write block if necessary.
	if p.wpend == nil {
		if !p.wclosed {
			p.wpend = <-p.cr
		}
		if p.wclosed {
			return 0, p.werr
		}
		p.wtot = 0
	}

	// Read from current write block.
	n = len(data)
	if n > len(p.wpend) {
		n = len(p.wpend)
	}
	for i := 0; i < n; i++ {
		data[i] = p.wpend[i]
	}
	p.wtot += n
	p.wpend = p.wpend[n:]

	// If write block is done, finish the write.
	if len(p.wpend) == 0 {
		p.wpend = nil
		p.cw <- pipeReturn{p.wtot, nil}
		p.wtot = 0
	}

	return n, nil
}

func (p *pipe) Write(data []byte) (n int, err os.Error) {
	if p == nil || p.wclosed {
		return 0, os.EINVAL
	}
	if p.rclosed {
		return 0, p.rerr
	}

	// Send data to reader.
	p.cr <- data

	// Wait for reader to finish copying it.
	res := <-p.cw
	return res.n, res.err
}

func (p *pipe) CloseReader(rerr os.Error) os.Error {
	if p == nil || p.rclosed {
		return os.EINVAL
	}

	// Stop any future writes.
	p.rclosed = true
	if rerr == nil {
		rerr = os.EPIPE
	}
	p.rerr = rerr

	// Stop the current write.
	if !p.wclosed {
		p.cw <- pipeReturn{p.wtot, rerr}
	}

	return nil
}

func (p *pipe) CloseWriter(werr os.Error) os.Error {
	if werr == nil {
		werr = os.EOF
	}
	if p == nil || p.wclosed {
		return os.EINVAL
	}

	// Stop any future reads.
	p.wclosed = true
	p.werr = werr

	// Stop the current read.
	if !p.rclosed {
		p.cr <- nil
	}

	return nil
}

// Read/write halves of the pipe.
// They are separate structures for two reasons:
//  1.  If one end becomes garbage without being Closed,
//      its finisher can Close so that the other end
//      does not hang indefinitely.
//  2.  Clients cannot use interface conversions on the
//      read end to find the Write method, and vice versa.

// A PipeReader is the read half of a pipe.
type PipeReader struct {
	lock sync.Mutex
	p    *pipe
}

// Read implements the standard Read interface:
// it reads data from the pipe, blocking until a writer
// arrives or the write end is closed.
// If the write end is closed with an error, that error is
// returned as err; otherwise err is nil.
func (r *PipeReader) Read(data []byte) (n int, err os.Error) {
	r.lock.Lock()
	defer r.lock.Unlock()

	return r.p.Read(data)
}

// Close closes the reader; subsequent writes to the
// write half of the pipe will return the error os.EPIPE.
func (r *PipeReader) Close() os.Error {
	r.lock.Lock()
	defer r.lock.Unlock()

	return r.p.CloseReader(nil)
}

// CloseWithError closes the reader; subsequent writes
// to the write half of the pipe will return the error rerr.
func (r *PipeReader) CloseWithError(rerr os.Error) os.Error {
	r.lock.Lock()
	defer r.lock.Unlock()

	return r.p.CloseReader(rerr)
}

func (r *PipeReader) finish() { r.Close() }

// Write half of pipe.
type PipeWriter struct {
	lock sync.Mutex
	p    *pipe
}

// Write implements the standard Write interface:
// it writes data to the pipe, blocking until readers
// have consumed all the data or the read end is closed.
// If the read end is closed with an error, that err is
// returned as err; otherwise err is os.EPIPE.
func (w *PipeWriter) Write(data []byte) (n int, err os.Error) {
	w.lock.Lock()
	defer w.lock.Unlock()

	return w.p.Write(data)
}

// Close closes the writer; subsequent reads from the
// read half of the pipe will return no bytes and a nil error.
func (w *PipeWriter) Close() os.Error {
	w.lock.Lock()
	defer w.lock.Unlock()

	return w.p.CloseWriter(nil)
}

// CloseWithError closes the writer; subsequent reads from the
// read half of the pipe will return no bytes and the error werr.
func (w *PipeWriter) CloseWithError(werr os.Error) os.Error {
	w.lock.Lock()
	defer w.lock.Unlock()

	return w.p.CloseWriter(werr)
}

func (w *PipeWriter) finish() { w.Close() }

// Pipe creates a synchronous in-memory pipe.
// It can be used to connect code expecting an io.Reader
// with code expecting an io.Writer.
// Reads on one end are matched with writes on the other,
// copying data directly between the two; there is no internal buffering.
func Pipe() (*PipeReader, *PipeWriter) {
	p := new(pipe)
	p.cr = make(chan []byte, 1)
	p.cw = make(chan pipeReturn, 1)
	r := new(PipeReader)
	r.p = p
	w := new(PipeWriter)
	w.p = p
	return r, w
}
