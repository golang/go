// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rpc

import (
	"bufio"
	"gob"
	"http"
	"io"
	"log"
	"net"
	"os"
	"sync"
)

// Call represents an active RPC.
type Call struct {
	ServiceMethod string      // The name of the service and method to call.
	Args          interface{} // The argument to the function (*struct).
	Reply         interface{} // The reply from the function (*struct).
	Error         os.Error    // After completion, the error status.
	Done          chan *Call  // Strobes when call is complete; value is the error status.
	seq           uint64
}

// Client represents an RPC Client.
// There may be multiple outstanding Calls associated
// with a single Client.
type Client struct {
	mutex    sync.Mutex // protects pending, seq
	shutdown os.Error   // non-nil if the client is shut down
	sending  sync.Mutex
	seq      uint64
	conn     io.ReadWriteCloser
	enc      *gob.Encoder
	dec      *gob.Decoder
	pending  map[uint64]*Call
}

func (client *Client) send(c *Call) {
	// Register this call.
	client.mutex.Lock()
	if client.shutdown != nil {
		c.Error = client.shutdown
		client.mutex.Unlock()
		_ = c.Done <- c // do not block
		return
	}
	c.seq = client.seq
	client.seq++
	client.pending[c.seq] = c
	client.mutex.Unlock()

	// Encode and send the request.
	request := new(Request)
	client.sending.Lock()
	request.Seq = c.seq
	request.ServiceMethod = c.ServiceMethod
	client.enc.Encode(request)
	err := client.enc.Encode(c.Args)
	if err != nil {
		panicln("rpc: client encode error:", err.String())
	}
	client.sending.Unlock()
}

func (client *Client) input() {
	var err os.Error
	for err == nil {
		response := new(Response)
		err = client.dec.Decode(response)
		if err != nil {
			if err == os.EOF {
				err = io.ErrUnexpectedEOF
			}
			break
		}
		seq := response.Seq
		client.mutex.Lock()
		c := client.pending[seq]
		client.pending[seq] = c, false
		client.mutex.Unlock()
		err = client.dec.Decode(c.Reply)
		// Empty strings should turn into nil os.Errors
		if response.Error != "" {
			c.Error = os.ErrorString(response.Error)
		} else {
			c.Error = nil
		}
		// We don't want to block here.  It is the caller's responsibility to make
		// sure the channel has enough buffer space. See comment in Go().
		_ = c.Done <- c // do not block
	}
	// Terminate pending calls.
	client.mutex.Lock()
	client.shutdown = err
	for _, call := range client.pending {
		call.Error = err
		_ = call.Done <- call // do not block
	}
	client.mutex.Unlock()
	log.Stderr("rpc: client protocol error:", err)
}

// NewClient returns a new Client to handle requests to the
// set of services at the other end of the connection.
func NewClient(conn io.ReadWriteCloser) *Client {
	client := new(Client)
	client.conn = conn
	client.enc = gob.NewEncoder(conn)
	client.dec = gob.NewDecoder(conn)
	client.pending = make(map[uint64]*Call)
	go client.input()
	return client
}

// DialHTTP connects to an HTTP RPC server at the specified network address.
func DialHTTP(network, address string) (*Client, os.Error) {
	conn, err := net.Dial(network, "", address)
	if err != nil {
		return nil, err
	}
	io.WriteString(conn, "CONNECT "+rpcPath+" HTTP/1.0\n\n")

	// Require successful HTTP response
	// before switching to RPC protocol.
	resp, err := http.ReadResponse(bufio.NewReader(conn), "CONNECT")
	if err == nil && resp.Status == connected {
		return NewClient(conn), nil
	}
	if err == nil {
		err = os.ErrorString("unexpected HTTP response: " + resp.Status)
	}
	conn.Close()
	return nil, &net.OpError{"dial-http", network + " " + address, nil, err}
}

// Dial connects to an RPC server at the specified network address.
func Dial(network, address string) (*Client, os.Error) {
	conn, err := net.Dial(network, "", address)
	if err != nil {
		return nil, err
	}
	return NewClient(conn), nil
}

// Go invokes the function asynchronously.  It returns the Call structure representing
// the invocation.  The done channel will signal when the call is complete by returning
// the same Call object.  If done is nil, Go will allocate a new channel.
// If non-nil, done must be buffered or Go will deliberately crash.
func (client *Client) Go(serviceMethod string, args interface{}, reply interface{}, done chan *Call) *Call {
	c := new(Call)
	c.ServiceMethod = serviceMethod
	c.Args = args
	c.Reply = reply
	if done == nil {
		done = make(chan *Call, 10) // buffered.
	} else {
		// If caller passes done != nil, it must arrange that
		// done has enough buffer for the number of simultaneous
		// RPCs that will be using that channel.  If the channel
		// is totally unbuffered, it's best not to run at all.
		if cap(done) == 0 {
			log.Crash("rpc: done channel is unbuffered")
		}
	}
	c.Done = done
	if client.shutdown != nil {
		c.Error = client.shutdown
		_ = c.Done <- c // do not block
		return c
	}
	client.send(c)
	return c
}

// Call invokes the named function, waits for it to complete, and returns its error status.
func (client *Client) Call(serviceMethod string, args interface{}, reply interface{}) os.Error {
	if client.shutdown != nil {
		return client.shutdown
	}
	call := <-client.Go(serviceMethod, args, reply, nil).Done
	return call.Error
}
