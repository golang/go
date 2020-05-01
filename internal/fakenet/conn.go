// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fakenet

import (
	"io"
	"net"
	"sync"
	"time"
)

// NewConn returns a net.Conn built on top of the supplied reader and writer.
// It decouples the read and write on the conn from the underlying stream
// to enable Close to abort ones that are in progress.
// It's primary use is to fake a network connection from stdin and stdout.
func NewConn(name string, in io.ReadCloser, out io.WriteCloser) net.Conn {
	c := &fakeConn{
		name:   name,
		reader: newFeeder(in.Read),
		writer: newFeeder(out.Write),
		in:     in,
		out:    out,
	}
	go c.reader.run()
	go c.writer.run()
	return c
}

type fakeConn struct {
	name   string
	reader *connFeeder
	writer *connFeeder
	in     io.ReadCloser
	out    io.WriteCloser
}

type fakeAddr string

// connFeeder serializes calls to the source function (io.Reader.Read or
// io.Writer.Write) by delegating them to a channel. This also allows calls to
// be intercepted when the connection is closed, and cancelled early if the
// connection is closed while the calls are still outstanding.
type connFeeder struct {
	source func([]byte) (int, error)
	input  chan []byte
	result chan feedResult
	mu     sync.Mutex
	closed bool
	done   chan struct{}
}

type feedResult struct {
	n   int
	err error
}

func (c *fakeConn) Close() error {
	c.reader.close()
	c.writer.close()
	c.in.Close()
	c.out.Close()
	return nil
}

func (c *fakeConn) Read(b []byte) (n int, err error)   { return c.reader.do(b) }
func (c *fakeConn) Write(b []byte) (n int, err error)  { return c.writer.do(b) }
func (c *fakeConn) LocalAddr() net.Addr                { return fakeAddr(c.name) }
func (c *fakeConn) RemoteAddr() net.Addr               { return fakeAddr(c.name) }
func (c *fakeConn) SetDeadline(t time.Time) error      { return nil }
func (c *fakeConn) SetReadDeadline(t time.Time) error  { return nil }
func (c *fakeConn) SetWriteDeadline(t time.Time) error { return nil }
func (a fakeAddr) Network() string                     { return "fake" }
func (a fakeAddr) String() string                      { return string(a) }

func newFeeder(source func([]byte) (int, error)) *connFeeder {
	return &connFeeder{
		source: source,
		input:  make(chan []byte),
		result: make(chan feedResult),
		done:   make(chan struct{}),
	}
}

func (f *connFeeder) close() {
	f.mu.Lock()
	if !f.closed {
		f.closed = true
		close(f.done)
	}
	f.mu.Unlock()
}

func (f *connFeeder) do(b []byte) (n int, err error) {
	// send the request to the worker
	select {
	case f.input <- b:
	case <-f.done:
		return 0, io.EOF
	}
	// get the result from the worker
	select {
	case r := <-f.result:
		return r.n, r.err
	case <-f.done:
		return 0, io.EOF
	}
}

func (f *connFeeder) run() {
	var b []byte
	for {
		// wait for an input request
		select {
		case b = <-f.input:
		case <-f.done:
			return
		}
		// invoke the underlying method
		n, err := f.source(b)
		// send the result back to the requester
		select {
		case f.result <- feedResult{n: n, err: err}:
		case <-f.done:
			return
		}
	}
}
