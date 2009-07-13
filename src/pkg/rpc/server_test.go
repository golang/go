// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rpc

import (
	"fmt";
	"gob";
	"http";
	"io";
	"log";
	"net";
	"os";
	"rpc";
	"testing";
)

var serverAddr string

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

func run(server *Server, l net.Listener) {
	conn, addr, err := l.Accept();
	if err != nil {
		println("accept:", err.String());
		os.Exit(1);
	}
	server.Serve(conn);
}

func startServer() {
	server := new(Server);
	server.Add(new(Arith));
	l, e := net.Listen("tcp", ":0");	// any available address
	if e != nil {
		log.Stderrf("net.Listen tcp :0: %v", e);
		os.Exit(1);
	}
	serverAddr = l.Addr();
	log.Stderr("Test RPC server listening on ", serverAddr);
//	go http.Serve(l, nil);
	go run(server, l);
}

func TestRPC(t *testing.T) {
	var i int;

	startServer();

	conn, err := net.Dial("tcp", "", serverAddr);
	if err != nil {
		t.Fatal("dialing:", err)
	}

	client := NewClient(conn);

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
	mulCall := client.Start("Arith.Mul", args, mulReply, nil);
	addReply := new(Reply);
	addCall := client.Start("Arith.Add", args, addReply, nil);

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
		t.Errorf("Div: expected error");
	}
}
