// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements Native Client's simple RPC (SRPC).
package srpc

import (
	"bytes";
	"log";
	"os";
	"sync";
)

// A Client represents the client side of an SRPC connection.
type Client struct {
	fd	int;	// fd to server
	r	msgReceiver;
	s	msgSender;
	service	map[string]srv;	// services by name
	out	chan *msg;	// send to out to write to connection

	mu	sync.Mutex;	// protects pending, idGen
	pending	map[uint64]*RPC;
	idGen	uint64;	// generator for request IDs
}

// A srv is a single method that the server offers.
type srv struct {
	num	uint32;	// method number
	fmt	string;	// argument format
}

// An RPC represents a single RPC issued by a client.
type RPC struct {
	Ret	[]interface{};	// Return values
	Done	chan *RPC;	// Channel where notification of done arrives
	Errno	Errno;		// Status code
	c	*Client;
	id	uint64;	// request id
}

// NewClient allocates a new client using the file descriptor fd.
func NewClient(fd int) (c *Client, err os.Error) {
	c = new(Client);
	c.fd = fd;
	c.r.fd = fd;
	c.s.fd = fd;
	c.service = make(map[string]srv);
	c.pending = make(map[uint64]*RPC);

	// service discovery request
	m := &msg{
		protocol: protocol,
		isReq: true,
		Ret: []interface{}{[]byte(nil)},
		Size: []int{4000},
	};
	m.packRequest();
	c.s.send(m);
	m, err = c.r.recv();
	if err != nil {
		return nil, err
	}
	m.unpackResponse();
	if m.status != OK {
		log.Stderrf("NewClient service_discovery: %s", m.status);
		return nil, m.status;
	}
	for n, line := range bytes.Split(m.Ret[0].([]byte), []byte{'\n'}, 0) {
		i := bytes.Index(line, []byte{':'});
		if i < 0 {
			continue
		}
		c.service[string(line[0:i])] = srv{uint32(n), string(line[i+1:])};
	}

	c.out = make(chan *msg);
	go c.input();
	go c.output();
	return c, nil;
}

func (c *Client) input() {
	for {
		m, err := c.r.recv();
		if err != nil {
			log.Exitf("client recv: %s", err)
		}
		if m.unpackResponse(); m.status != OK {
			log.Stderrf("invalid message: %s", m.status);
			continue;
		}
		c.mu.Lock();
		rpc, ok := c.pending[m.requestId];
		if ok {
			c.pending[m.requestId] = nil, false
		}
		c.mu.Unlock();
		if !ok {
			log.Stderrf("unexpected response");
			continue;
		}
		rpc.Ret = m.Ret;
		rpc.Done <- rpc;
	}
}

func (c *Client) output() {
	for m := range c.out {
		c.s.send(m)
	}
}

// NewRPC creates a new RPC on the client connection.
func (c *Client) NewRPC(done chan *RPC) *RPC {
	if done == nil {
		done = make(chan *RPC)
	}
	c.mu.Lock();
	id := c.idGen;
	c.idGen++;
	c.mu.Unlock();
	return &RPC{nil, done, OK, c, id};
}

// Start issues an RPC request for method name with the given arguments.
// The RPC r must not be in use for another pending request.
// To wait for the RPC to finish, receive from r.Done and then
// inspect r.Ret and r.Errno.
func (r *RPC) Start(name string, arg []interface{}) {
	var m msg;

	r.Errno = OK;
	r.c.mu.Lock();
	srv, ok := r.c.service[name];
	if !ok {
		r.c.mu.Unlock();
		r.Errno = ErrBadRPCNumber;
		r.Done <- r;
		return;
	}
	r.c.pending[r.id] = r;
	r.c.mu.Unlock();

	m.protocol = protocol;
	m.requestId = r.id;
	m.isReq = true;
	m.rpcNumber = srv.num;
	m.Arg = arg;

	// Fill in the return values and sizes to generate
	// the right type chars.  We'll take most any size.

	// Skip over input arguments.
	// We could check them against arg, but the server
	// will do that anyway.
	i := 0;
	for srv.fmt[i] != ':' {
		i++
	}
	fmt := srv.fmt[i+1:];

	// Now the return prototypes.
	m.Ret = make([]interface{}, len(fmt)-i);
	m.Size = make([]int, len(fmt)-i);
	for i := 0; i < len(fmt); i++ {
		switch fmt[i] {
		default:
			log.Exitf("unexpected service type %c", fmt[i])
		case 'b':
			m.Ret[i] = false
		case 'C':
			m.Ret[i] = []byte(nil);
			m.Size[i] = 1 << 30;
		case 'd':
			m.Ret[i] = float64(0)
		case 'D':
			m.Ret[i] = []float64(nil);
			m.Size[i] = 1 << 30;
		case 'h':
			m.Ret[i] = int(-1)
		case 'i':
			m.Ret[i] = int32(0)
		case 'I':
			m.Ret[i] = []int32(nil);
			m.Size[i] = 1 << 30;
		case 's':
			m.Ret[i] = "";
			m.Size[i] = 1 << 30;
		}
	}

	m.packRequest();
	r.c.out <- &m;
}

// Call is a convenient wrapper that starts the RPC request,
// waits for it to finish, and then returns the results.
// Its implementation is:
//
//	r.Start(name, arg);
//	<-r.Done;
//	return r.Ret, r.Errno;
//
func (r *RPC) Call(name string, arg []interface{}) (ret []interface{}, err Errno) {
	r.Start(name, arg);
	<-r.Done;
	return r.Ret, r.Errno;
}
