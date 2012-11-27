// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"runtime"
	"testing"
	"time"
)

func TestNoRaceChanSync(t *testing.T) {
	v := 0
	c := make(chan int)
	go func() {
		v = 1
		c <- 0
	}()
	<-c
	v = 2
}

func TestNoRaceChanSyncRev(t *testing.T) {
	v := 0
	c := make(chan int)
	go func() {
		c <- 0
		v = 2
	}()
	v = 1
	<-c
}

func TestNoRaceChanAsync(t *testing.T) {
	v := 0
	c := make(chan int, 10)
	go func() {
		v = 1
		c <- 0
	}()
	<-c
	v = 2
}

func TestRaceChanAsyncRev(t *testing.T) {
	v := 0
	c := make(chan int, 10)
	go func() {
		c <- 0
		v = 1
	}()
	v = 2
	<-c
}

func TestNoRaceChanAsyncCloseRecv(t *testing.T) {
	v := 0
	c := make(chan int, 10)
	go func() {
		v = 1
		close(c)
	}()
	func() {
		defer func() {
			recover()
			v = 2
		}()
		<-c
	}()
}

func TestNoRaceChanAsyncCloseRecv2(t *testing.T) {
	v := 0
	c := make(chan int, 10)
	go func() {
		v = 1
		close(c)
	}()
	_, _ = <-c
	v = 2
}

func TestNoRaceChanAsyncCloseRecv3(t *testing.T) {
	v := 0
	c := make(chan int, 10)
	go func() {
		v = 1
		close(c)
	}()
	for _ = range c {
	}
	v = 2
}

func TestNoRaceChanSyncCloseRecv(t *testing.T) {
	v := 0
	c := make(chan int)
	go func() {
		v = 1
		close(c)
	}()
	func() {
		defer func() {
			recover()
			v = 2
		}()
		<-c
	}()
}

func TestNoRaceChanSyncCloseRecv2(t *testing.T) {
	v := 0
	c := make(chan int)
	go func() {
		v = 1
		close(c)
	}()
	_, _ = <-c
	v = 2
}

func TestNoRaceChanSyncCloseRecv3(t *testing.T) {
	v := 0
	c := make(chan int)
	go func() {
		v = 1
		close(c)
	}()
	for _ = range c {
	}
	v = 2
}

func TestRaceChanSyncCloseSend(t *testing.T) {
	v := 0
	c := make(chan int)
	go func() {
		v = 1
		close(c)
	}()
	func() {
		defer func() {
			recover()
		}()
		c <- 0
	}()
	v = 2
}

func TestRaceChanAsyncCloseSend(t *testing.T) {
	v := 0
	c := make(chan int, 10)
	go func() {
		v = 1
		close(c)
	}()
	func() {
		defer func() {
			recover()
		}()
		for {
			c <- 0
		}
	}()
	v = 2
}

func TestRaceChanCloseClose(t *testing.T) {
	compl := make(chan bool, 2)
	v1 := 0
	v2 := 0
	c := make(chan int)
	go func() {
		defer func() {
			if recover() != nil {
				v2 = 2
			}
			compl <- true
		}()
		v1 = 1
		close(c)
	}()
	go func() {
		defer func() {
			if recover() != nil {
				v1 = 2
			}
			compl <- true
		}()
		v2 = 1
		close(c)
	}()
	<-compl
	<-compl
}

func TestRaceChanSendLen(t *testing.T) {
	v := 0
	c := make(chan int, 10)
	go func() {
		v = 1
		c <- 1
	}()
	for len(c) == 0 {
		runtime.Gosched()
	}
	v = 2
}

func TestRaceChanRecvLen(t *testing.T) {
	v := 0
	c := make(chan int, 10)
	c <- 1
	go func() {
		v = 1
		<-c
	}()
	for len(c) != 0 {
		runtime.Gosched()
	}
	v = 2
}

func TestRaceChanSendSend(t *testing.T) {
	compl := make(chan bool, 2)
	v1 := 0
	v2 := 0
	c := make(chan int, 1)
	go func() {
		v1 = 1
		select {
		case c <- 1:
		default:
			v2 = 2
		}
		compl <- true
	}()
	go func() {
		v2 = 1
		select {
		case c <- 1:
		default:
			v1 = 2
		}
		compl <- true
	}()
	<-compl
	<-compl
}

func TestNoRaceChanPtr(t *testing.T) {
	type msg struct {
		x int
	}
	c := make(chan *msg)
	go func() {
		c <- &msg{1}
	}()
	m := <-c
	m.x = 2
}

func TestRaceChanWrongSend(t *testing.T) {
	v1 := 0
	v2 := 0
	c := make(chan int, 2)
	go func() {
		v1 = 1
		c <- 1
	}()
	go func() {
		v2 = 2
		c <- 2
	}()
	time.Sleep(1e7)
	if <-c == 1 {
		v2 = 3
	} else {
		v1 = 3
	}
}

func TestRaceChanWrongClose(t *testing.T) {
	v1 := 0
	v2 := 0
	c := make(chan int, 1)
	go func() {
		defer func() {
			recover()
		}()
		v1 = 1
		c <- 1
	}()
	go func() {
		time.Sleep(1e7)
		v2 = 2
		close(c)
	}()
	time.Sleep(2e7)
	if _, who := <-c; who {
		v2 = 2
	} else {
		v1 = 2
	}
}

func TestRaceChanSendClose(t *testing.T) {
	compl := make(chan bool, 2)
	c := make(chan int, 1)
	go func() {
		defer func() {
			recover()
		}()
		c <- 1
		compl <- true
	}()
	go func() {
		time.Sleep(1e7)
		close(c)
		compl <- true
	}()
	<-compl
	<-compl
}

func TestNoRaceProducerConsumerUnbuffered(t *testing.T) {
	type Task struct {
		f    func()
		done chan bool
	}

	queue := make(chan Task)

	go func() {
		t := <-queue
		t.f()
		t.done <- true
	}()

	doit := func(f func()) {
		done := make(chan bool, 1)
		queue <- Task{f, done}
		<-done
	}

	x := 0
	doit(func() {
		x = 1
	})
	_ = x
}

func TestRaceChanItselfSend(t *testing.T) {
	compl := make(chan bool, 1)
	c := make(chan int, 10)
	go func() {
		c <- 0
		compl <- true
	}()
	c = make(chan int, 20)
	<-compl
}

func TestRaceChanItselfRecv(t *testing.T) {
	compl := make(chan bool, 1)
	c := make(chan int, 10)
	c <- 1
	go func() {
		<-c
		compl <- true
	}()
	time.Sleep(1e7)
	c = make(chan int, 20)
	<-compl
}

func TestRaceChanItselfNil(t *testing.T) {
	c := make(chan int, 10)
	go func() {
		c <- 0
	}()
	time.Sleep(1e7)
	c = nil
	_ = c
}

func TestRaceChanItselfClose(t *testing.T) {
	compl := make(chan bool, 1)
	c := make(chan int)
	go func() {
		close(c)
		compl <- true
	}()
	c = make(chan int)
	<-compl
}

func TestRaceChanItselfLen(t *testing.T) {
	compl := make(chan bool, 1)
	c := make(chan int)
	go func() {
		_ = len(c)
		compl <- true
	}()
	c = make(chan int)
	<-compl
}

func TestRaceChanItselfCap(t *testing.T) {
	compl := make(chan bool, 1)
	c := make(chan int)
	go func() {
		_ = cap(c)
		compl <- true
	}()
	c = make(chan int)
	<-compl
}

func TestRaceChanCloseLen(t *testing.T) {
	v := 0
	c := make(chan int, 10)
	c <- 0
	go func() {
		v = 1
		close(c)
	}()
	time.Sleep(1e7)
	_ = len(c)
	v = 2
}

func TestRaceChanSameCell(t *testing.T) {
	c := make(chan int, 1)
	v := 0
	go func() {
		v = 1
		c <- 42
		<-c
	}()
	time.Sleep(1e7)
	c <- 43
	<-c
	_ = v
}

func TestRaceChanCloseSend(t *testing.T) {
	compl := make(chan bool, 1)
	c := make(chan int, 10)
	go func() {
		close(c)
		compl <- true
	}()
	c <- 0
	<-compl
}
