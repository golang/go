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

// ServerError represents an error that has been returned from
// the remote side of the RPC connection.
type ServerError string

func (e ServerError) String() string {
	return string(e)
}

var ErrShutdown = os.NewError("connection is shut down")

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
	mutex    sync.Mutex // protects pending, seq, request
	sending  sync.Mutex
	request  Request
	seq      uint64
	codec    ClientCodec
	pending  map[uint64]*Call
	closing  bool
	shutdown bool
}

// A ClientCodec implements writing of RPC requests and
// reading of RPC responses for the client side of an RPC session.
// The client calls WriteRequest to write a request to the connection
// and calls ReadResponseHeader and ReadResponseBody in pairs
// to read responses.  The client calls Close when finished with the
// connection. ReadResponseBody may be called with a nil
// argument to force the body of the response to be read and then
// discarded.
type ClientCodec interface {
	WriteRequest(*Request, interface{}) os.Error
	ReadResponseHeader(*Response) os.Error
	ReadResponseBody(interface{}) os.Error

	Close() os.Error
}

func (client *Client) send(c *Call) {
	// Register this call.
	client.mutex.Lock()
	if client.shutdown {
		c.Error = ErrShutdown
		client.mutex.Unlock()
		c.done()
		return
	}
	c.seq = client.seq
	client.seq++
	client.pending[c.seq] = c
	client.mutex.Unlock()

	// Encode and send the request.
	client.sending.Lock()
	defer client.sending.Unlock()
	client.request.Seq = c.seq
	client.request.ServiceMethod = c.ServiceMethod
	if err := client.codec.WriteRequest(&client.request, c.Args); err != nil {
		c.Error = err
		c.done()
	}
}

func (client *Client) input() {
	var err os.Error
	var response Response
	for err == nil {
		response = Response{}
		err = client.codec.ReadResponseHeader(&response)
		if err != nil {
			if err == os.EOF && !client.closing {
				err = io.ErrUnexpectedEOF
			}
			break
		}
		seq := response.Seq
		client.mutex.Lock()
		c := client.pending[seq]
		delete(client.pending, seq)
		client.mutex.Unlock()

		if response.Error == "" {
			err = client.codec.ReadResponseBody(c.Reply)
			if err != nil {
				c.Error = os.NewError("reading body " + err.String())
			}
		} else {
			// We've got an error response. Give this to the request;
			// any subsequent requests will get the ReadResponseBody
			// error if there is one.
			c.Error = ServerError(response.Error)
			err = client.codec.ReadResponseBody(nil)
			if err != nil {
				err = os.NewError("reading error body: " + err.String())
			}
		}
		c.done()
	}
	// Terminate pending calls.
	client.mutex.Lock()
	client.shutdown = true
	for _, call := range client.pending {
		call.Error = err
		call.done()
	}
	client.mutex.Unlock()
	if err != os.EOF || !client.closing {
		log.Println("rpc: client protocol error:", err)
	}
}

func (call *Call) done() {
	select {
	case call.Done <- call:
		// ok
	default:
		// We don't want to block here.  It is the caller's responsibility to make
		// sure the channel has enough buffer space. See comment in Go().
	}
}

// NewClient returns a new Client to handle requests to the
// set of services at the other end of the connection.
// It adds a buffer to the write side of the connection so
// the header and payload are sent as a unit.
func NewClient(conn io.ReadWriteCloser) *Client {
	encBuf := bufio.NewWriter(conn)
	client := &gobClientCodec{conn, gob.NewDecoder(conn), gob.NewEncoder(encBuf), encBuf}
	return NewClientWithCodec(client)
}

// NewClientWithCodec is like NewClient but uses the specified
// codec to encode requests and decode responses.
func NewClientWithCodec(codec ClientCodec) *Client {
	client := &Client{
		codec:   codec,
		pending: make(map[uint64]*Call),
	}
	go client.input()
	return client
}

type gobClientCodec struct {
	rwc    io.ReadWriteCloser
	dec    *gob.Decoder
	enc    *gob.Encoder
	encBuf *bufio.Writer
}

func (c *gobClientCodec) WriteRequest(r *Request, body interface{}) (err os.Error) {
	if err = c.enc.Encode(r); err != nil {
		return
	}
	if err = c.enc.Encode(body); err != nil {
		return
	}
	return c.encBuf.Flush()
}

func (c *gobClientCodec) ReadResponseHeader(r *Response) os.Error {
	return c.dec.Decode(r)
}

func (c *gobClientCodec) ReadResponseBody(body interface{}) os.Error {
	return c.dec.Decode(body)
}

func (c *gobClientCodec) Close() os.Error {
	return c.rwc.Close()
}

// DialHTTP connects to an HTTP RPC server at the specified network address
// listening on the default HTTP RPC path.
func DialHTTP(network, address string) (*Client, os.Error) {
	return DialHTTPPath(network, address, DefaultRPCPath)
}

// DialHTTPPath connects to an HTTP RPC server 
// at the specified network address and path.
func DialHTTPPath(network, address, path string) (*Client, os.Error) {
	var err os.Error
	conn, err := net.Dial(network, address)
	if err != nil {
		return nil, err
	}
	io.WriteString(conn, "CONNECT "+path+" HTTP/1.0\n\n")

	// Require successful HTTP response
	// before switching to RPC protocol.
	resp, err := http.ReadResponse(bufio.NewReader(conn), &http.Request{Method: "CONNECT"})
	if err == nil && resp.Status == connected {
		return NewClient(conn), nil
	}
	if err == nil {
		err = os.NewError("unexpected HTTP response: " + resp.Status)
	}
	conn.Close()
	return nil, &net.OpError{"dial-http", network + " " + address, nil, err}
}

// Dial connects to an RPC server at the specified network address.
func Dial(network, address string) (*Client, os.Error) {
	conn, err := net.Dial(network, address)
	if err != nil {
		return nil, err
	}
	return NewClient(conn), nil
}

func (client *Client) Close() os.Error {
	client.mutex.Lock()
	if client.shutdown || client.closing {
		client.mutex.Unlock()
		return ErrShutdown
	}
	client.closing = true
	client.mutex.Unlock()
	return client.codec.Close()
}

// Go invokes the function asynchronously.  It returns the Call structure representing
// the invocation.  The done channel will signal when the call is complete by returning
// the same Call object.  If done is nil, Go will allocate a new channel.
// If non-nil, done must be buffered or Go will deliberately crash.
func (client *Client) Go(serviceMethod string, args interface{}, reply interface{}, done chan *Call) *Call {
	call := new(Call)
	call.ServiceMethod = serviceMethod
	call.Args = args
	call.Reply = reply
	if done == nil {
		done = make(chan *Call, 10) // buffered.
	} else {
		// If caller passes done != nil, it must arrange that
		// done has enough buffer for the number of simultaneous
		// RPCs that will be using that channel.  If the channel
		// is totally unbuffered, it's best not to run at all.
		if cap(done) == 0 {
			log.Panic("rpc: done channel is unbuffered")
		}
	}
	call.Done = done
	if client.shutdown {
		call.Error = ErrShutdown
		call.done()
		return call
	}
	client.send(call)
	return call
}

// Call invokes the named function, waits for it to complete, and returns its error status.
func (client *Client) Call(serviceMethod string, args interface{}, reply interface{}) os.Error {
	if client.shutdown {
		return ErrShutdown
	}
	call := <-client.Go(serviceMethod, args, reply, make(chan *Call, 1)).Done
	return call.Error
}
