// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Request struct {
	a, b	int;
	replyc	*chan int;
}

type BinOp (a, b int) int;

func Run(op *BinOp, request *Request) {
	result := op(request.a, request.b);
	request.replyc <- result;
}

func Server(op *BinOp, service *chan *Request) {
	for {
		request := <-service;
		go Run(op, request);  // don't wait for it
	}
}

func StartServer(op *BinOp) *chan *Request {
	req := new(chan *Request);
	go Server(op, req);
	return req;
}

func main() {
	adder := StartServer(func(a, b int) int { return a + b });
	const N = 100;
	var reqs [N]Request;
	for i := 0; i < N; i++ {
		req := &reqs[i];
		req.a = i;
		req.b = i + N;
		req.replyc = new(chan int);
		adder <- req;
	}
	for i := N-1; i >= 0; i-- {   // doesn't matter what order
		if <-reqs[i].replyc != N + 2*i {
			print("fail at ", i, "\n");
		}
	}
	print("done\n");
}
