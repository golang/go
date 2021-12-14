// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rpc

import (
	"errors"
	"fmt"
	"net"
	"strings"
	"testing"
)

type shutdownCodec struct {
	responded chan int
	closed    bool
}

func (c *shutdownCodec) WriteRequest(*Request, interface{}) error { return nil }
func (c *shutdownCodec) ReadResponseBody(interface{}) error       { return nil }
func (c *shutdownCodec) ReadResponseHeader(*Response) error {
	c.responded <- 1
	return errors.New("shutdownCodec ReadResponseHeader")
}
func (c *shutdownCodec) Close() error {
	c.closed = true
	return nil
}

func TestCloseCodec(t *testing.T) {
	codec := &shutdownCodec{responded: make(chan int)}
	client := NewClientWithCodec(codec)
	<-codec.responded
	client.Close()
	if !codec.closed {
		t.Error("client.Close did not close codec")
	}
}

// Test that errors in gob shut down the connection. Issue 7689.

type R struct {
	msg []byte // Not exported, so R does not work with gob.
}

type S struct{}

func (s *S) Recv(nul *struct{}, reply *R) error {
	*reply = R{[]byte("foo")}
	return nil
}

func TestGobError(t *testing.T) {
	defer func() {
		err := recover()
		if err == nil {
			t.Fatal("no error")
		}
		if !strings.Contains(err.(error).Error(), "reading body unexpected EOF") {
			t.Fatal("expected `reading body unexpected EOF', got", err)
		}
	}()
	Register(new(S))

	listen, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	go Accept(listen)

	client, err := Dial("tcp", listen.Addr().String())
	if err != nil {
		panic(err)
	}

	var reply Reply
	err = client.Call("S.Recv", &struct{}{}, &reply)
	if err != nil {
		panic(err)
	}

	fmt.Printf("%#v\n", reply)
	client.Close()

	listen.Close()
}
