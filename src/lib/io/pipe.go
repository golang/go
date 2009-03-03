// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Pipe adapter to connect code expecting an io.Read
// with code expecting an io.Write.

package io

import (
	"io";
	"os";
	"sync";
)

type pipeReturn struct {
	n int;
	err *os.Error;
}

// Shared pipe structure.
type pipe struct {
	rclosed bool;		// Read end closed?
	wclosed bool;		// Write end closed?
	wpend []byte;		// Written data waiting to be read.
	wtot int;		// Bytes consumed so far in current write.
	cr chan []byte;		// Write sends data here...
	cw chan pipeReturn;	// ... and reads the n, err back from here.
}

func (p *pipe) Read(data []byte) (n int, err *os.Error) {
	if p == nil || p.rclosed {
		return 0, os.EINVAL;
	}

	// Wait for next write block if necessary.
	if p.wpend == nil {
		if !p.wclosed {
			p.wpend = <-p.cr;
		}
		if p.wpend == nil {
			return 0, nil;
		}
		p.wtot = 0;
	}

	// Read from current write block.
	n = len(data);
	if n > len(p.wpend) {
		n = len(p.wpend);
	}
	for i := 0; i < n; i++ {
		data[i] = p.wpend[i];
	}
	p.wtot += n;
	p.wpend = p.wpend[n:len(p.wpend)];

	// If write block is done, finish the write.
	if len(p.wpend) == 0 {
		p.wpend = nil;
		p.cw <- pipeReturn{p.wtot, nil};
		p.wtot = 0;
	}

	return n, nil;
}

func (p *pipe) Write(data []byte) (n int, err *os.Error) {
	if p == nil || p.wclosed {
		return 0, os.EINVAL;
	}
	if p.rclosed {
		return 0, os.EPIPE;
	}

	// Send data to reader.
	p.cr <- data;

	// Wait for reader to finish copying it.
	res := <-p.cw;
	return res.n, res.err;
}

func (p *pipe) CloseReader() *os.Error {
	if p == nil || p.rclosed {
		return os.EINVAL;
	}

	// Stop any future writes.
	p.rclosed = true;

	// Stop the current write.
	if !p.wclosed {
		p.cw <- pipeReturn{p.wtot, os.EPIPE};
	}

	return nil;
}

func (p *pipe) CloseWriter() *os.Error {
	if p == nil || p.wclosed {
		return os.EINVAL;
	}

	// Stop any future reads.
	p.wclosed = true;

	// Stop the current read.
	if !p.rclosed {
		p.cr <- nil;
	}

	return nil;
}

// Read/write halves of the pipe.
// They are separate structures for two reasons:
//  1.  If one end becomes garbage without being Closed,
//      its finisher can Close so that the other end
//      does not hang indefinitely.
//  2.  Clients cannot use interface conversions on the
//      read end to find the Write method, and vice versa.

// Read half of pipe.
type pipeRead struct {
	lock sync.Mutex;
	p *pipe;
}

func (r *pipeRead) Read(data []byte) (n int, err *os.Error) {
	r.lock.Lock();
	defer r.lock.Unlock();

	return r.p.Read(data);
}

func (r *pipeRead) Close() *os.Error {
	r.lock.Lock();
	defer r.lock.Unlock();

	return r.p.CloseReader();
}

func (r *pipeRead) finish() {
	r.Close();
}

// Write half of pipe.
type pipeWrite struct {
	lock sync.Mutex;
	p *pipe;
}

func (w *pipeWrite) Write(data []byte) (n int, err *os.Error) {
	w.lock.Lock();
	defer w.lock.Unlock();

	return w.p.Write(data);
}

func (w *pipeWrite) Close() *os.Error {
	w.lock.Lock();
	defer w.lock.Unlock();

	return w.p.CloseWriter();
}

func (w *pipeWrite) finish() {
	w.Close();
}

// Create a synchronous in-memory pipe.
// Reads on one end are matched by writes on the other.
// Writes don't complete until all the data has been
// written or the read end is closed.  Reads return
// any available data or block until the next write
// or the write end is closed.
func Pipe() (io.ReadClose, io.WriteClose) {
	p := new(pipe);
	p.cr = make(chan []byte, 1);
	p.cw = make(chan pipeReturn, 1);
	r := new(pipeRead);
	r.p = p;
	w := new(pipeWrite);
	w.p = p;
	return r, w;
}

