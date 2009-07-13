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

	enc := gob.NewEncoder(conn);
	dec := gob.NewDecoder(conn);
	req := new(rpc.Request);
	req.ServiceMethod = "Arith.Add";
	req.Seq = 1;
	enc.Encode(req);
	args := &Args{7,8};
	enc.Encode(args);
	response := new(rpc.Response);
	dec.Decode(response);
	reply := new(Reply);
	dec.Decode(reply);
	fmt.Printf("%d\n", reply.C);
	if reply.C != args.A + args.B {
		t.Errorf("Add: expected %d got %d", reply.C != args.A + args.B);
	}

	req.ServiceMethod = "Arith.Mul";
	req.Seq++;
	enc.Encode(req);
	args = &Args{7,8};
	enc.Encode(args);
	response = new(rpc.Response);
	dec.Decode(response);
	reply = new(Reply);
	dec.Decode(reply);
	fmt.Printf("%d\n", reply.C);
	if reply.C != args.A * args.B {
		t.Errorf("Mul: expected %d got %d", reply.C != args.A * args.B);
	}

	req.ServiceMethod = "Arith.Div";
	req.Seq++;
	enc.Encode(req);
	args = &Args{7,0};
	enc.Encode(args);
	response = new(rpc.Response);
	dec.Decode(response);
	reply = new(Reply);
	dec.Decode(reply);
	// expect an error: zero divide
	if response.Error == "" {
		t.Errorf("Div: expected error");
	}
}

