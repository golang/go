// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rpc

import (
	"gob";
	"io";
	"os";
	"rpc";
	"sync";
)

// Call represents an active RPC
type Call struct {
	ServiceMethod	string;	// The name of the service and method to call.
	Args	interface{};	// The argument to the function (*struct).
	Reply	interface{};	// The reply from the function (*struct).
	Error	os.Error;	// After completion, the error status.
	Done	chan *Call;	// Strobes when call is complete; value is the error status.
	seq	uint64;
}

// Client represents an RPC Client.
type Client struct {
	sync.Mutex;	// protects pending, seq
	sending	sync.Mutex;
	seq	uint64;
	conn io.ReadWriteCloser;
	enc	*gob.Encoder;
	dec	*gob.Decoder;
	pending	map[uint64] *Call;
}

func (client *Client) send(c *Call) {
	// Register this call.
	client.Lock();
	c.seq = client.seq;
	client.seq++;
	client.pending[c.seq] = c;
	client.Unlock();

	// Encode and send the request.
	request := new(Request);
	client.sending.Lock();
	request.Seq = c.seq;
	request.ServiceMethod = c.ServiceMethod;
	client.enc.Encode(request);
	err := client.enc.Encode(c.Args);
	if err != nil {
		panicln("client encode error:", err)
	}
	client.sending.Unlock();
}

func (client *Client) serve() {
	for {
		response := new(Response);
		err := client.dec.Decode(response);
		seq := response.Seq;
		client.Lock();
		c := client.pending[seq];
		client.pending[seq] = c, false;
		client.Unlock();
		client.dec.Decode(c.Reply);
		if err != nil {
			panicln("client decode error:", err)
		}
		c.Error = os.ErrorString(response.Error);
		// We don't want to block here, it is the caller's responsibility to make
		// sure the channel has enough buffer space. See comment in Start().
		doNotBlock := c.Done <- c;
	}
}

// NewClient returns a new Client to handle requests to the
// set of services at the other end of the connection.
func NewClient(conn io.ReadWriteCloser) *Client {
	client := new(Client);
	client.conn = conn;
	client.enc = gob.NewEncoder(conn);
	client.dec = gob.NewDecoder(conn);
	client.pending = make(map[uint64] *Call);
	go client.serve();
	return client;
}

// Start invokes the function asynchronously.  It returns the Call structure representing
// the invocation.
func (client *Client) Start(serviceMethod string, args interface{}, reply interface{}, done chan *Call) *Call {
	c := new(Call);
	c.ServiceMethod = serviceMethod;
	c.Args = args;
	c.Reply = reply;
	if done == nil {
		done = make(chan *Call, 1);	// buffered.
	} else {
		// TODO(r): check cap > 0
		// If caller passes done != nil, it must arrange that
		// done has enough buffer for the number of simultaneous
		// RPCs that will be using that channel.
	}
	c.Done = done;
	client.send(c);
	return c;
}

// Call invokes the named function, waits for it to complete, and returns its error status.
func (client *Client) Call(serviceMethod string, args interface{}, reply interface{}) os.Error {
	call := <-client.Start(serviceMethod, args, reply, nil).Done;
	return call.Error;
}
