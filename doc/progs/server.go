// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type request struct {
	a, b    int;
	replyc  chan int;
}

type binOp (a, b int) int;

func run(op *BinOp, request *Request) {
	result := op(request.a, request.b);
	request.replyc <- result;
}

func server(op *BinOp, service chan *Request) {
	for {
		request := <-service;
		go run(op, request);  // don't wait for it
	}
}

func startServer(op *BinOp) chan *Request {
	req := make(chan *Request);
	go Server(op, req);
	return req;
}

func main() {
	adder := startServer(func(a, b int) int { return a + b });
	const N = 100;
	var reqs [N]request;
	for i := 0; i < N; i++ {
		req := &reqs[i];
		req.a = i;
		req.b = i + N;
		req.replyc = make(chan int);
		adder <- req;
	}
	for i := N-1; i >= 0; i-- {   // doesn't matter what order
		if <-reqs[i].replyc != N + 2*i {
			print("fail at ", i, "\n");
		}
	}
	print("done\n");
}
