// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"io"
	"time"
)

// Pipe creates a synchronous, in-memory, full duplex
// network connection; both ends implement the Conn interface.
// Reads on one end are matched with writes on the other,
// copying data directly between the two; there is no internal
// buffering.
func Pipe() (Conn, Conn) {
	r1, w1 := io.Pipe()
	r2, w2 := io.Pipe()

	return &pipe{r1, w2}, &pipe{r2, w1}
}

type pipe struct {
	*io.PipeReader
	*io.PipeWriter
}

type pipeAddr int

func (pipeAddr) Network() string {
	return "pipe"
}

func (pipeAddr) String() string {
	return "pipe"
}

func (p *pipe) Close() error {
	err := p.PipeReader.Close()
	err1 := p.PipeWriter.Close()
	if err == nil {
		err = err1
	}
	return err
}

func (p *pipe) LocalAddr() Addr {
	return pipeAddr(0)
}

func (p *pipe) RemoteAddr() Addr {
	return pipeAddr(0)
}

func (p *pipe) SetDeadline(t time.Time) error {
	return errors.New("net.Pipe does not support deadlines")
}

func (p *pipe) SetReadDeadline(t time.Time) error {
	return errors.New("net.Pipe does not support deadlines")
}

func (p *pipe) SetWriteDeadline(t time.Time) error {
	return errors.New("net.Pipe does not support deadlines")
}
