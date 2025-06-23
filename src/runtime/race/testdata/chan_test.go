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
	_ = v
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
	_ = v
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
	_ = v
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
	_ = v
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
	_ = v
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
	_ = v
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
	_ = v
	c := make(chan int, 10)
	go func() {
		v = 1
		close(c)
	}()
	for range c {
	}
	v = 2
}

func TestNoRaceChanSyncCloseRecv(t *testing.T) {
	v := 0
	_ = v
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
	_ = v
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
	_ = v
	c := make(chan int)
	go func() {
		v = 1
		close(c)
	}()
	for range c {
	}
	v = 2
}

func TestRaceChanSyncCloseSend(t *testing.T) {
	v := 0
	_ = v
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
	_ = v
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
	_ = v1 + v2
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
	_ = v
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
	_ = v
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
	_ = v1 + v2
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
	_ = v1 + v2
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
	_ = v1 + v2
	c := make(chan int, 1)
	done := make(chan bool)
	go func() {
		defer func() {
			recover()
		}()
		v1 = 1
		c <- 1
		done <- true
	}()
	go func() {
		time.Sleep(1e7)
		v2 = 2
		close(c)
		done <- true
	}()
	time.Sleep(2e7)
	if _, who := <-c; who {
		v2 = 2
	} else {
		v1 = 2
	}
	<-done
	<-done
}

func TestRaceChanSendClose(t *testing.T) {
	compl := make(chan bool, 2)
	c := make(chan int, 1)
	go func() {
		defer func() {
			recover()
			compl <- true
		}()
		c <- 1
	}()
	go func() {
		time.Sleep(10 * time.Millisecond)
		close(c)
		compl <- true
	}()
	<-compl
	<-compl
}

func TestRaceChanSendSelectClose(t *testing.T) {
	compl := make(chan bool, 2)
	c := make(chan int, 1)
	c1 := make(chan int)
	go func() {
		defer func() {
			recover()
			compl <- true
		}()
		time.Sleep(10 * time.Millisecond)
		select {
		case c <- 1:
		case <-c1:
		}
	}()
	go func() {
		close(c)
		compl <- true
	}()
	<-compl
	<-compl
}

func TestRaceSelectReadWriteAsync(t *testing.T) {
	done := make(chan bool)
	x := 0
	c1 := make(chan int, 10)
	c2 := make(chan int, 10)
	c3 := make(chan int)
	c2 <- 1
	go func() {
		select {
		case c1 <- x: // read of x races with...
		case c3 <- 1:
		}
		done <- true
	}()
	select {
	case x = <-c2: // ... write to x here
	case c3 <- 1:
	}
	<-done
}

func TestRaceSelectReadWriteSync(t *testing.T) {
	done := make(chan bool)
	x := 0
	c1 := make(chan int)
	c2 := make(chan int)
	c3 := make(chan int)
	// make c1 and c2 ready for communication
	go func() {
		<-c1
	}()
	go func() {
		c2 <- 1
	}()
	go func() {
		select {
		case c1 <- x: // read of x races with...
		case c3 <- 1:
		}
		done <- true
	}()
	select {
	case x = <-c2: // ... write to x here
	case c3 <- 1:
	}
	<-done
}

func TestNoRaceSelectReadWriteAsync(t *testing.T) {
	done := make(chan bool)
	x := 0
	c1 := make(chan int)
	c2 := make(chan int)
	go func() {
		select {
		case c1 <- x: // read of x does not race with...
		case c2 <- 1:
		}
		done <- true
	}()
	select {
	case x = <-c1: // ... write to x here
	case c2 <- 1:
	}
	<-done
}

func TestRaceChanReadWriteAsync(t *testing.T) {
	done := make(chan bool)
	c1 := make(chan int, 10)
	c2 := make(chan int, 10)
	c2 <- 10
	x := 0
	go func() {
		c1 <- x // read of x races with...
		done <- true
	}()
	x = <-c2 // ... write to x here
	<-done
}

func TestRaceChanReadWriteSync(t *testing.T) {
	done := make(chan bool)
	c1 := make(chan int)
	c2 := make(chan int)
	// make c1 and c2 ready for communication
	go func() {
		<-c1
	}()
	go func() {
		c2 <- 10
	}()
	x := 0
	go func() {
		c1 <- x // read of x races with...
		done <- true
	}()
	x = <-c2 // ... write to x here
	<-done
}

func TestNoRaceChanReadWriteAsync(t *testing.T) {
	done := make(chan bool)
	c1 := make(chan int, 10)
	x := 0
	go func() {
		c1 <- x // read of x does not race with...
		done <- true
	}()
	x = <-c1 // ... write to x here
	<-done
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

func TestNoRaceChanCloseLen(t *testing.T) {
	c := make(chan int, 10)
	r := make(chan int, 10)
	go func() {
		r <- len(c)
	}()
	go func() {
		close(c)
		r <- 0
	}()
	<-r
	<-r
}

func TestNoRaceChanCloseCap(t *testing.T) {
	c := make(chan int, 10)
	r := make(chan int, 10)
	go func() {
		r <- cap(c)
	}()
	go func() {
		close(c)
		r <- 0
	}()
	<-r
	<-r
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

func TestNoRaceChanMutex(t *testing.T) {
	done := make(chan struct{})
	mtx := make(chan struct{}, 1)
	data := 0
	_ = data
	go func() {
		mtx <- struct{}{}
		data = 42
		<-mtx
		done <- struct{}{}
	}()
	mtx <- struct{}{}
	data = 43
	<-mtx
	<-done
}

func TestNoRaceSelectMutex(t *testing.T) {
	done := make(chan struct{})
	mtx := make(chan struct{}, 1)
	aux := make(chan bool)
	data := 0
	_ = data
	go func() {
		select {
		case mtx <- struct{}{}:
		case <-aux:
		}
		data = 42
		select {
		case <-mtx:
		case <-aux:
		}
		done <- struct{}{}
	}()
	select {
	case mtx <- struct{}{}:
	case <-aux:
	}
	data = 43
	select {
	case <-mtx:
	case <-aux:
	}
	<-done
}

func TestRaceChanSem(t *testing.T) {
	done := make(chan struct{})
	mtx := make(chan bool, 2)
	data := 0
	_ = data
	go func() {
		mtx <- true
		data = 42
		<-mtx
		done <- struct{}{}
	}()
	mtx <- true
	data = 43
	<-mtx
	<-done
}

func TestNoRaceChanWaitGroup(t *testing.T) {
	const N = 10
	chanWg := make(chan bool, N/2)
	data := make([]int, N)
	for i := 0; i < N; i++ {
		chanWg <- true
		go func(i int) {
			data[i] = 42
			<-chanWg
		}(i)
	}
	for i := 0; i < cap(chanWg); i++ {
		chanWg <- true
	}
	for i := 0; i < N; i++ {
		_ = data[i]
	}
}

// Test that sender synchronizes with receiver even if the sender was blocked.
func TestNoRaceBlockedSendSync(t *testing.T) {
	c := make(chan *int, 1)
	c <- nil
	go func() {
		i := 42
		c <- &i
	}()
	// Give the sender time to actually block.
	// This sleep is completely optional: race report must not be printed
	// regardless of whether the sender actually blocks or not.
	// It cannot lead to flakiness.
	time.Sleep(10 * time.Millisecond)
	<-c
	p := <-c
	if *p != 42 {
		t.Fatal()
	}
}

// The same as TestNoRaceBlockedSendSync above, but sender unblock happens in a select.
func TestNoRaceBlockedSelectSendSync(t *testing.T) {
	c := make(chan *int, 1)
	c <- nil
	go func() {
		i := 42
		c <- &i
	}()
	time.Sleep(10 * time.Millisecond)
	<-c
	select {
	case p := <-c:
		if *p != 42 {
			t.Fatal()
		}
	case <-make(chan int):
	}
}

// Test that close synchronizes with a read from the empty closed channel.
// See https://golang.org/issue/36714.
func TestNoRaceCloseHappensBeforeRead(t *testing.T) {
	for i := 0; i < 100; i++ {
		var loc int
		var write = make(chan struct{})
		var read = make(chan struct{})

		go func() {
			select {
			case <-write:
				_ = loc
			default:
			}
			close(read)
		}()

		go func() {
			loc = 1
			close(write)
		}()

		<-read
	}
}

// Test that we call the proper race detector function when c.elemsize==0.
// See https://github.com/golang/go/issues/42598
func TestNoRaceElemSize0(t *testing.T) {
	var x, y int
	var c = make(chan struct{}, 2)
	c <- struct{}{}
	c <- struct{}{}
	go func() {
		x += 1
		<-c
	}()
	go func() {
		y += 1
		<-c
	}()
	time.Sleep(10 * time.Millisecond)
	c <- struct{}{}
	c <- struct{}{}
	x += 1
	y += 1
}
