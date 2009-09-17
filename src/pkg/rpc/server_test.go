// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rpc

import (
	"http";
	"log";
	"net";
	"once";
	"os";
	"strings";
	"testing";
)

var serverAddr string
var httpServerAddr string

const second = 1e9


type Args struct {
	A, B int
}

type Reply struct {
	C int
}

type Arith int

func (t *Arith) Add(args *Args, reply *Reply) os.Error {
	reply.C = args.A + args.B;
	return nil
}

func (t *Arith) Mul(args *Args, reply *Reply) os.Error {
	reply.C = args.A * args.B;
	return nil
}

func (t *Arith) Div(args *Args, reply *Reply) os.Error {
	if args.B == 0 {
		return os.ErrorString("divide by zero");
	}
	reply.C = args.A / args.B;
	return nil
}

func (t *Arith) Error(args *Args, reply *Reply) os.Error {
	panicln("ERROR");
}

func startServer() {
	Register(new(Arith));

	l, e := net.Listen("tcp", ":0");	// any available address
	if e != nil {
		log.Exitf("net.Listen tcp :0: %v", e);
	}
	serverAddr = l.Addr();
	log.Stderr("Test RPC server listening on ", serverAddr);
	go Accept(l);

	HandleHTTP();
	l, e = net.Listen("tcp", ":0");	// any available address
	if e != nil {
		log.Stderrf("net.Listen tcp :0: %v", e);
		os.Exit(1);
	}
	httpServerAddr = l.Addr();
	log.Stderr("Test HTTP RPC server listening on ", httpServerAddr);
	go http.Serve(l, nil);
}

func TestRPC(t *testing.T) {
	once.Do(startServer);

	client, err := Dial("tcp", serverAddr);
	if err != nil {
		t.Fatal("dialing", err);
	}

	// Synchronous calls
	args := &Args{7,8};
	reply := new(Reply);
	err = client.Call("Arith.Add", args, reply);
	if reply.C != args.A + args.B {
		t.Errorf("Add: expected %d got %d", reply.C, args.A + args.B);
	}

	args = &Args{7,8};
	reply = new(Reply);
	err = client.Call("Arith.Mul", args, reply);
	if reply.C != args.A * args.B {
		t.Errorf("Mul: expected %d got %d", reply.C, args.A * args.B);
	}

	// Out of order.
	args = &Args{7,8};
	mulReply := new(Reply);
	mulCall := client.Go("Arith.Mul", args, mulReply, nil);
	addReply := new(Reply);
	addCall := client.Go("Arith.Add", args, addReply, nil);

	<-addCall.Done;
	if addReply.C != args.A + args.B {
		t.Errorf("Add: expected %d got %d", addReply.C, args.A + args.B);
	}

	<-mulCall.Done;
	if mulReply.C != args.A * args.B {
		t.Errorf("Mul: expected %d got %d", mulReply.C, args.A * args.B);
	}

	// Error test
	args = &Args{7,0};
	reply = new(Reply);
	err = client.Call("Arith.Div", args, reply);
	// expect an error: zero divide
	if err == nil {
		t.Error("Div: expected error");
	} else if err.String() != "divide by zero" {
		t.Error("Div: expected divide by zero error; got", err);
	}
}

func TestHTTPRPC(t *testing.T) {
	once.Do(startServer);

	client, err := DialHTTP("tcp", httpServerAddr);
	if err != nil {
		t.Fatal("dialing", err);
	}

	// Synchronous calls
	args := &Args{7,8};
	reply := new(Reply);
	err = client.Call("Arith.Add", args, reply);
	if reply.C != args.A + args.B {
		t.Errorf("Add: expected %d got %d", reply.C, args.A + args.B);
	}
}

func TestCheckUnknownService(t *testing.T) {
	once.Do(startServer);

	conn, err := net.Dial("tcp", "", serverAddr);
	if err != nil {
		t.Fatal("dialing:", err)
	}

	client := NewClient(conn);

	args := &Args{7,8};
	reply := new(Reply);
	err = client.Call("Unknown.Add", args, reply);
	if err == nil {
		t.Error("expected error calling unknown service");
	} else if strings.Index(err.String(), "service") < 0 {
		t.Error("expected error about service; got", err);
	}
}

func TestCheckUnknownMethod(t *testing.T) {
	once.Do(startServer);

	conn, err := net.Dial("tcp", "", serverAddr);
	if err != nil {
		t.Fatal("dialing:", err)
	}

	client := NewClient(conn);

	args := &Args{7,8};
	reply := new(Reply);
	err = client.Call("Arith.Unknown", args, reply);
	if err == nil {
		t.Error("expected error calling unknown service");
	} else if strings.Index(err.String(), "method") < 0 {
		t.Error("expected error about method; got", err);
	}
}

func TestCheckBadType(t *testing.T) {
	once.Do(startServer);

	conn, err := net.Dial("tcp", "", serverAddr);
	if err != nil {
		t.Fatal("dialing:", err)
	}

	client := NewClient(conn);

	reply := new(Reply);
	err = client.Call("Arith.Add", reply, reply);	// args, reply would be the correct thing to use
	if err == nil {
		t.Error("expected error calling Arith.Add with wrong arg type");
	} else if strings.Index(err.String(), "type") < 0 {
		t.Error("expected error about type; got", err);
	}
}
