// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type request struct {
	a, b	int;
	replyc	chan int;
}

type binOp func(a, b int) int

func run(op binOp, req *request) {
	reply := op(req.a, req.b);
	req.replyc <- reply;
}

func server(op binOp, service chan *request, quit chan bool) {
	for {
		select {
		case req := <-service:
			go run(op, req);  // don't wait for it
		case <-quit:
			return;
		}
	}
}

func startServer(op binOp) (service chan *request, quit chan bool) {
	service = make(chan *request);
	quit = make(chan bool);
	go server(op, service, quit);
	return service, quit;
}

func main() {
	adder, quit := startServer(func(a, b int) int { return a + b });
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
			fmt.Println("fail at", i);
		}
	}
	quit <- true;
}
