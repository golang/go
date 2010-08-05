// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rpc

import (
	"fmt"
	"http"
	"log"
	"net"
	"os"
	"strings"
	"sync"
	"testing"
)

var serverAddr string
var httpServerAddr string
var once sync.Once

const second = 1e9


type Args struct {
	A, B int
}

type Reply struct {
	C int
}

type Arith int

func (t *Arith) Add(args *Args, reply *Reply) os.Error {
	reply.C = args.A + args.B
	return nil
}

func (t *Arith) Mul(args *Args, reply *Reply) os.Error {
	reply.C = args.A * args.B
	return nil
}

func (t *Arith) Div(args *Args, reply *Reply) os.Error {
	if args.B == 0 {
		return os.ErrorString("divide by zero")
	}
	reply.C = args.A / args.B
	return nil
}

func (t *Arith) String(args *Args, reply *string) os.Error {
	*reply = fmt.Sprintf("%d+%d=%d", args.A, args.B, args.A+args.B)
	return nil
}

func (t *Arith) Scan(args *string, reply *Reply) (err os.Error) {
	_, err = fmt.Sscan(*args, &reply.C)
	return
}

func (t *Arith) Error(args *Args, reply *Reply) os.Error {
	panic("ERROR")
}

func startServer() {
	Register(new(Arith))

	l, e := net.Listen("tcp", "127.0.0.1:0") // any available address
	if e != nil {
		log.Exitf("net.Listen tcp :0: %v", e)
	}
	serverAddr = l.Addr().String()
	log.Stderr("Test RPC server listening on ", serverAddr)
	go Accept(l)

	HandleHTTP()
	l, e = net.Listen("tcp", "127.0.0.1:0") // any available address
	if e != nil {
		log.Stderrf("net.Listen tcp :0: %v", e)
		os.Exit(1)
	}
	httpServerAddr = l.Addr().String()
	log.Stderr("Test HTTP RPC server listening on ", httpServerAddr)
	go http.Serve(l, nil)
}

func TestRPC(t *testing.T) {
	once.Do(startServer)

	client, err := Dial("tcp", serverAddr)
	if err != nil {
		t.Fatal("dialing", err)
	}

	// Synchronous calls
	args := &Args{7, 8}
	reply := new(Reply)
	err = client.Call("Arith.Add", args, reply)
	if err != nil {
		t.Errorf("Add: expected no error but got string %q", err.String())
	}
	if reply.C != args.A+args.B {
		t.Errorf("Add: expected %d got %d", reply.C, args.A+args.B)
	}

	args = &Args{7, 8}
	reply = new(Reply)
	err = client.Call("Arith.Mul", args, reply)
	if err != nil {
		t.Errorf("Mul: expected no error but got string %q", err.String())
	}
	if reply.C != args.A*args.B {
		t.Errorf("Mul: expected %d got %d", reply.C, args.A*args.B)
	}

	// Out of order.
	args = &Args{7, 8}
	mulReply := new(Reply)
	mulCall := client.Go("Arith.Mul", args, mulReply, nil)
	addReply := new(Reply)
	addCall := client.Go("Arith.Add", args, addReply, nil)

	addCall = <-addCall.Done
	if addCall.Error != nil {
		t.Errorf("Add: expected no error but got string %q", addCall.Error.String())
	}
	if addReply.C != args.A+args.B {
		t.Errorf("Add: expected %d got %d", addReply.C, args.A+args.B)
	}

	mulCall = <-mulCall.Done
	if mulCall.Error != nil {
		t.Errorf("Mul: expected no error but got string %q", mulCall.Error.String())
	}
	if mulReply.C != args.A*args.B {
		t.Errorf("Mul: expected %d got %d", mulReply.C, args.A*args.B)
	}

	// Error test
	args = &Args{7, 0}
	reply = new(Reply)
	err = client.Call("Arith.Div", args, reply)
	// expect an error: zero divide
	if err == nil {
		t.Error("Div: expected error")
	} else if err.String() != "divide by zero" {
		t.Error("Div: expected divide by zero error; got", err)
	}

	// Non-struct argument
	const Val = 12345
	str := fmt.Sprint(Val)
	reply = new(Reply)
	err = client.Call("Arith.Scan", &str, reply)
	if err != nil {
		t.Errorf("Scan: expected no error but got string %q", err.String())
	} else if reply.C != Val {
		t.Errorf("Scan: expected %d got %d", Val, reply.C)
	}

	// Non-struct reply
	args = &Args{27, 35}
	str = ""
	err = client.Call("Arith.String", args, &str)
	if err != nil {
		t.Errorf("String: expected no error but got string %q", err.String())
	}
	expect := fmt.Sprintf("%d+%d=%d", args.A, args.B, args.A+args.B)
	if str != expect {
		t.Errorf("String: expected %s got %s", expect, str)
	}
}

func TestHTTPRPC(t *testing.T) {
	once.Do(startServer)

	client, err := DialHTTP("tcp", httpServerAddr)
	if err != nil {
		t.Fatal("dialing", err)
	}

	// Synchronous calls
	args := &Args{7, 8}
	reply := new(Reply)
	err = client.Call("Arith.Add", args, reply)
	if err != nil {
		t.Errorf("Add: expected no error but got string %q", err.String())
	}
	if reply.C != args.A+args.B {
		t.Errorf("Add: expected %d got %d", reply.C, args.A+args.B)
	}
}

func TestCheckUnknownService(t *testing.T) {
	once.Do(startServer)

	conn, err := net.Dial("tcp", "", serverAddr)
	if err != nil {
		t.Fatal("dialing:", err)
	}

	client := NewClient(conn)

	args := &Args{7, 8}
	reply := new(Reply)
	err = client.Call("Unknown.Add", args, reply)
	if err == nil {
		t.Error("expected error calling unknown service")
	} else if strings.Index(err.String(), "service") < 0 {
		t.Error("expected error about service; got", err)
	}
}

func TestCheckUnknownMethod(t *testing.T) {
	once.Do(startServer)

	conn, err := net.Dial("tcp", "", serverAddr)
	if err != nil {
		t.Fatal("dialing:", err)
	}

	client := NewClient(conn)

	args := &Args{7, 8}
	reply := new(Reply)
	err = client.Call("Arith.Unknown", args, reply)
	if err == nil {
		t.Error("expected error calling unknown service")
	} else if strings.Index(err.String(), "method") < 0 {
		t.Error("expected error about method; got", err)
	}
}

func TestCheckBadType(t *testing.T) {
	once.Do(startServer)

	conn, err := net.Dial("tcp", "", serverAddr)
	if err != nil {
		t.Fatal("dialing:", err)
	}

	client := NewClient(conn)

	reply := new(Reply)
	err = client.Call("Arith.Add", reply, reply) // args, reply would be the correct thing to use
	if err == nil {
		t.Error("expected error calling Arith.Add with wrong arg type")
	} else if strings.Index(err.String(), "type") < 0 {
		t.Error("expected error about type; got", err)
	}
}

type ArgNotPointer int
type ReplyNotPointer int
type ArgNotPublic int
type ReplyNotPublic int
type local struct{}

func (t *ArgNotPointer) ArgNotPointer(args Args, reply *Reply) os.Error {
	return nil
}

func (t *ReplyNotPointer) ReplyNotPointer(args *Args, reply Reply) os.Error {
	return nil
}

func (t *ArgNotPublic) ArgNotPublic(args *local, reply *Reply) os.Error {
	return nil
}

func (t *ReplyNotPublic) ReplyNotPublic(args *Args, reply *local) os.Error {
	return nil
}

// Check that registration handles lots of bad methods and a type with no suitable methods.
func TestRegistrationError(t *testing.T) {
	err := Register(new(ArgNotPointer))
	if err == nil {
		t.Errorf("expected error registering ArgNotPointer")
	}
	err = Register(new(ReplyNotPointer))
	if err == nil {
		t.Errorf("expected error registering ReplyNotPointer")
	}
	err = Register(new(ArgNotPublic))
	if err == nil {
		t.Errorf("expected error registering ArgNotPublic")
	}
	err = Register(new(ReplyNotPublic))
	if err == nil {
		t.Errorf("expected error registering ReplyNotPublic")
	}
}
