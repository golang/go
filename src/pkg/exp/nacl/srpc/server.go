// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// SRPC server

package srpc

import (
	"bytes"
	"log"
	"os"
	"syscall"
)

// TODO(rsc): I'd prefer to make this
//	type Handler func(m *msg) Errno
// but NaCl can't use closures.
// The explicit interface is a way to attach state.

// A Handler is a handler for an SRPC method.
// It reads arguments from arg, checks size for array limits,
// writes return values to ret, and returns an Errno status code.
type Handler interface {
	Run(arg, ret []interface{}, size []int) Errno
}

type method struct {
	name    string
	fmt     string
	handler Handler
}

var rpcMethod []method

// BUG(rsc): Add's format string should be replaced by analyzing the
// type of an arbitrary func passed in an interface{} using reflection.

// Add registers a handler for the named method.
// Fmt is a Native Client format string, a sequence of
// alphabetic characters representing the types of the parameter values,
// a colon, and then a sequence of alphabetic characters
// representing the types of the returned values.
// The format characters and corresponding dynamic types are:
//
//	b	bool
//	C	[]byte
//	d	float64
//	D	[]float64
//	h	int	// a file descriptor (aka handle)
//	i	int32
//	I	[]int32
//	s	string
//
func Add(name, fmt string, handler Handler) {
	rpcMethod = append(rpcMethod, method{name, fmt, handler})
}

// Serve accepts new SRPC connections from the file descriptor fd
// and answers RPCs issued on those connections.
// It closes fd and returns an error if the imc_accept system call fails.
func Serve(fd int) os.Error {
	defer syscall.Close(fd)

	for {
		cfd, _, e := syscall.Syscall(syscall.SYS_IMC_ACCEPT, uintptr(fd), 0, 0)
		if e != 0 {
			return os.NewSyscallError("imc_accept", int(e))
		}
		go serveLoop(int(cfd))
	}
	panic("unreachable")
}

func serveLoop(fd int) {
	c := make(chan *msg)
	go sendLoop(fd, c)

	var r msgReceiver
	r.fd = fd
	for {
		m, err := r.recv()
		if err != nil {
			break
		}
		m.unpackRequest()
		if !m.gotHeader {
			log.Printf("cannot unpack header: %s", m.status)
			continue
		}
		// log.Printf("<- %#v", m);
		m.isReq = false // set up for response
		go serveMsg(m, c)
	}
	close(c)
}

func sendLoop(fd int, c <-chan *msg) {
	var s msgSender
	s.fd = fd
	for m := range c {
		// log.Printf("-> %#v", m);
		m.packResponse()
		s.send(m)
	}
	syscall.Close(fd)
}

func serveMsg(m *msg, c chan<- *msg) {
	if m.status != OK {
		c <- m
		return
	}
	if m.rpcNumber >= uint32(len(rpcMethod)) {
		m.status = ErrBadRPCNumber
		c <- m
		return
	}

	meth := &rpcMethod[m.rpcNumber]
	if meth.fmt != m.fmt {
		switch {
		case len(m.fmt) < len(meth.fmt):
			m.status = ErrTooFewArgs
		case len(m.fmt) > len(meth.fmt):
			m.status = ErrTooManyArgs
		default:
			// There's a type mismatch.
			// It's an in-arg mismatch if the mismatch happens
			// before the colon; otherwise it's an out-arg mismatch.
			m.status = ErrInArgTypeMismatch
			for i := 0; i < len(m.fmt) && m.fmt[i] == meth.fmt[i]; i++ {
				if m.fmt[i] == ':' {
					m.status = ErrOutArgTypeMismatch
					break
				}
			}
		}
		c <- m
		return
	}

	m.status = meth.handler.Run(m.Arg, m.Ret, m.Size)
	c <- m
}

// ServeRuntime serves RPCs issued by the Native Client embedded runtime.
// This should be called by main once all methods have been registered using Add.
func ServeRuntime() os.Error {
	// Call getFd to check that we are running embedded.
	if _, err := getFd(); err != nil {
		return err
	}

	// We are running embedded.
	// The fd returned by getFd is a red herring.
	// Accept connections on magic fd 3.
	return Serve(3)
}

// getFd runs the srpc_get_fd system call.
func getFd() (fd int, err os.Error) {
	r1, _, e := syscall.Syscall(syscall.SYS_SRPC_GET_FD, 0, 0, 0)
	return int(r1), os.NewSyscallError("srpc_get_fd", int(e))
}

// Enabled returns true if SRPC is enabled in the Native Client runtime.
func Enabled() bool {
	_, err := getFd()
	return err == nil
}

// Service #0, service_discovery, returns a list of the other services
// and their argument formats.
type serviceDiscovery struct{}

func (serviceDiscovery) Run(arg, ret []interface{}, size []int) Errno {
	var b bytes.Buffer
	for _, m := range rpcMethod {
		b.WriteString(m.name)
		b.WriteByte(':')
		b.WriteString(m.fmt)
		b.WriteByte('\n')
	}
	if b.Len() > size[0] {
		return ErrNoMemory
	}
	ret[0] = b.Bytes()
	return OK
}

func init() { Add("service_discovery", ":C", serviceDiscovery{}) }
