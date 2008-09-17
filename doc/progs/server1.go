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

func Server(op *BinOp, service *chan *Request, quit *chan bool) {
	for {
		select {
		case request := <-service:
			go Run(op, request);  // don't wait for it
		case <-quit:
			return;
		}
	}
}

func StartServer(op *BinOp) (servch *chan *Request, quitch *chan bool) {
	service := new(chan *Request);
	quit := new(chan bool);
	go Server(op, service, quit);
	return service, quit;
}

func main() {
	adder, quit := StartServer(func(a, b int) int { return a + b });
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
	quit <- true;
}
