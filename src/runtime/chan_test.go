// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/testenv"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestChan(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	N := 200
	if testing.Short() {
		N = 20
	}
	for chanCap := 0; chanCap < N; chanCap++ {
		{
			// Ensure that receive from empty chan blocks.
			c := make(chan int, chanCap)
			recv1 := false
			go func() {
				_ = <-c
				recv1 = true
			}()
			recv2 := false
			go func() {
				_, _ = <-c
				recv2 = true
			}()
			time.Sleep(time.Millisecond)
			if recv1 || recv2 {
				t.Fatalf("chan[%d]: receive from empty chan", chanCap)
			}
			// Ensure that non-blocking receive does not block.
			select {
			case _ = <-c:
				t.Fatalf("chan[%d]: receive from empty chan", chanCap)
			default:
			}
			select {
			case _, _ = <-c:
				t.Fatalf("chan[%d]: receive from empty chan", chanCap)
			default:
			}
			c <- 0
			c <- 0
		}

		{
			// Ensure that send to full chan blocks.
			c := make(chan int, chanCap)
			for i := 0; i < chanCap; i++ {
				c <- i
			}
			sent := uint32(0)
			go func() {
				c <- 0
				atomic.StoreUint32(&sent, 1)
			}()
			time.Sleep(time.Millisecond)
			if atomic.LoadUint32(&sent) != 0 {
				t.Fatalf("chan[%d]: send to full chan", chanCap)
			}
			// Ensure that non-blocking send does not block.
			select {
			case c <- 0:
				t.Fatalf("chan[%d]: send to full chan", chanCap)
			default:
			}
			<-c
		}

		{
			// Ensure that we receive 0 from closed chan.
			c := make(chan int, chanCap)
			for i := 0; i < chanCap; i++ {
				c <- i
			}
			close(c)
			for i := 0; i < chanCap; i++ {
				v := <-c
				if v != i {
					t.Fatalf("chan[%d]: received %v, expected %v", chanCap, v, i)
				}
			}
			if v := <-c; v != 0 {
				t.Fatalf("chan[%d]: received %v, expected %v", chanCap, v, 0)
			}
			if v, ok := <-c; v != 0 || ok {
				t.Fatalf("chan[%d]: received %v/%v, expected %v/%v", chanCap, v, ok, 0, false)
			}
		}

		{
			// Ensure that close unblocks receive.
			c := make(chan int, chanCap)
			done := make(chan bool)
			go func() {
				v, ok := <-c
				done <- v == 0 && ok == false
			}()
			time.Sleep(time.Millisecond)
			close(c)
			if !<-done {
				t.Fatalf("chan[%d]: received non zero from closed chan", chanCap)
			}
		}

		{
			// Send 100 integers,
			// ensure that we receive them non-corrupted in FIFO order.
			c := make(chan int, chanCap)
			go func() {
				for i := 0; i < 100; i++ {
					c <- i
				}
			}()
			for i := 0; i < 100; i++ {
				v := <-c
				if v != i {
					t.Fatalf("chan[%d]: received %v, expected %v", chanCap, v, i)
				}
			}

			// Same, but using recv2.
			go func() {
				for i := 0; i < 100; i++ {
					c <- i
				}
			}()
			for i := 0; i < 100; i++ {
				v, ok := <-c
				if !ok {
					t.Fatalf("chan[%d]: receive failed, expected %v", chanCap, i)
				}
				if v != i {
					t.Fatalf("chan[%d]: received %v, expected %v", chanCap, v, i)
				}
			}

			// Send 1000 integers in 4 goroutines,
			// ensure that we receive what we send.
			const P = 4
			const L = 1000
			for p := 0; p < P; p++ {
				go func() {
					for i := 0; i < L; i++ {
						c <- i
					}
				}()
			}
			done := make(chan map[int]int)
			for p := 0; p < P; p++ {
				go func() {
					recv := make(map[int]int)
					for i := 0; i < L; i++ {
						v := <-c
						recv[v] = recv[v] + 1
					}
					done <- recv
				}()
			}
			recv := make(map[int]int)
			for p := 0; p < P; p++ {
				for k, v := range <-done {
					recv[k] = recv[k] + v
				}
			}
			if len(recv) != L {
				t.Fatalf("chan[%d]: received %v values, expected %v", chanCap, len(recv), L)
			}
			for _, v := range recv {
				if v != P {
					t.Fatalf("chan[%d]: received %v values, expected %v", chanCap, v, P)
				}
			}
		}

		{
			// Test len/cap.
			c := make(chan int, chanCap)
			if len(c) != 0 || cap(c) != chanCap {
				t.Fatalf("chan[%d]: bad len/cap, expect %v/%v, got %v/%v", chanCap, 0, chanCap, len(c), cap(c))
			}
			for i := 0; i < chanCap; i++ {
				c <- i
			}
			if len(c) != chanCap || cap(c) != chanCap {
				t.Fatalf("chan[%d]: bad len/cap, expect %v/%v, got %v/%v", chanCap, chanCap, chanCap, len(c), cap(c))
			}
		}

	}
}

func TestNonblockRecvRace(t *testing.T) {
	n := 10000
	if testing.Short() {
		n = 100
	}
	for i := 0; i < n; i++ {
		c := make(chan int, 1)
		c <- 1
		go func() {
			select {
			case <-c:
			default:
				t.Error("chan is not ready")
			}
		}()
		close(c)
		<-c
		if t.Failed() {
			return
		}
	}
}

// This test checks that select acts on the state of the channels at one
// moment in the execution, not over a smeared time window.
// In the test, one goroutine does:
//
//	create c1, c2
//	make c1 ready for receiving
//	create second goroutine
//	make c2 ready for receiving
//	make c1 no longer ready for receiving (if possible)
//
// The second goroutine does a non-blocking select receiving from c1 and c2.
// From the time the second goroutine is created, at least one of c1 and c2
// is always ready for receiving, so the select in the second goroutine must
// always receive from one or the other. It must never execute the default case.
func TestNonblockSelectRace(t *testing.T) {
	n := 100000
	if testing.Short() {
		n = 1000
	}
	done := make(chan bool, 1)
	for i := 0; i < n; i++ {
		c1 := make(chan int, 1)
		c2 := make(chan int, 1)
		c1 <- 1
		go func() {
			select {
			case <-c1:
			case <-c2:
			default:
				done <- false
				return
			}
			done <- true
		}()
		c2 <- 1
		select {
		case <-c1:
		default:
		}
		if !<-done {
			t.Fatal("no chan is ready")
		}
	}
}

// Same as TestNonblockSelectRace, but close(c2) replaces c2 <- 1.
func TestNonblockSelectRace2(t *testing.T) {
	n := 100000
	if testing.Short() {
		n = 1000
	}
	done := make(chan bool, 1)
	for i := 0; i < n; i++ {
		c1 := make(chan int, 1)
		c2 := make(chan int)
		c1 <- 1
		go func() {
			select {
			case <-c1:
			case <-c2:
			default:
				done <- false
				return
			}
			done <- true
		}()
		close(c2)
		select {
		case <-c1:
		default:
		}
		if !<-done {
			t.Fatal("no chan is ready")
		}
	}
}

func TestSelfSelect(t *testing.T) {
	// Ensure that send/recv on the same chan in select
	// does not crash nor deadlock.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(2))
	for _, chanCap := range []int{0, 10} {
		var wg sync.WaitGroup
		wg.Add(2)
		c := make(chan int, chanCap)
		for p := 0; p < 2; p++ {
			p := p
			go func() {
				defer wg.Done()
				for i := 0; i < 1000; i++ {
					if p == 0 || i%2 == 0 {
						select {
						case c <- p:
						case v := <-c:
							if chanCap == 0 && v == p {
								t.Errorf("self receive")
								return
							}
						}
					} else {
						select {
						case v := <-c:
							if chanCap == 0 && v == p {
								t.Errorf("self receive")
								return
							}
						case c <- p:
						}
					}
				}
			}()
		}
		wg.Wait()
	}
}

func TestSelectStress(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(10))
	var c [4]chan int
	c[0] = make(chan int)
	c[1] = make(chan int)
	c[2] = make(chan int, 2)
	c[3] = make(chan int, 3)
	N := int(1e5)
	if testing.Short() {
		N /= 10
	}
	// There are 4 goroutines that send N values on each of the chans,
	// + 4 goroutines that receive N values on each of the chans,
	// + 1 goroutine that sends N values on each of the chans in a single select,
	// + 1 goroutine that receives N values on each of the chans in a single select.
	// All these sends, receives and selects interact chaotically at runtime,
	// but we are careful that this whole construct does not deadlock.
	var wg sync.WaitGroup
	wg.Add(10)
	for k := 0; k < 4; k++ {
		k := k
		go func() {
			for i := 0; i < N; i++ {
				c[k] <- 0
			}
			wg.Done()
		}()
		go func() {
			for i := 0; i < N; i++ {
				<-c[k]
			}
			wg.Done()
		}()
	}
	go func() {
		var n [4]int
		c1 := c
		for i := 0; i < 4*N; i++ {
			select {
			case c1[3] <- 0:
				n[3]++
				if n[3] == N {
					c1[3] = nil
				}
			case c1[2] <- 0:
				n[2]++
				if n[2] == N {
					c1[2] = nil
				}
			case c1[0] <- 0:
				n[0]++
				if n[0] == N {
					c1[0] = nil
				}
			case c1[1] <- 0:
				n[1]++
				if n[1] == N {
					c1[1] = nil
				}
			}
		}
		wg.Done()
	}()
	go func() {
		var n [4]int
		c1 := c
		for i := 0; i < 4*N; i++ {
			select {
			case <-c1[0]:
				n[0]++
				if n[0] == N {
					c1[0] = nil
				}
			case <-c1[1]:
				n[1]++
				if n[1] == N {
					c1[1] = nil
				}
			case <-c1[2]:
				n[2]++
				if n[2] == N {
					c1[2] = nil
				}
			case <-c1[3]:
				n[3]++
				if n[3] == N {
					c1[3] = nil
				}
			}
		}
		wg.Done()
	}()
	wg.Wait()
}

func TestSelectFairness(t *testing.T) {
	const trials = 10000
	if runtime.GOOS == "linux" && runtime.GOARCH == "ppc64le" {
		testenv.SkipFlaky(t, 22047)
	}
	c1 := make(chan byte, trials+1)
	c2 := make(chan byte, trials+1)
	for i := 0; i < trials+1; i++ {
		c1 <- 1
		c2 <- 2
	}
	c3 := make(chan byte)
	c4 := make(chan byte)
	out := make(chan byte)
	done := make(chan byte)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			var b byte
			select {
			case b = <-c3:
			case b = <-c4:
			case b = <-c1:
			case b = <-c2:
			}
			select {
			case out <- b:
			case <-done:
				return
			}
		}
	}()
	cnt1, cnt2 := 0, 0
	for i := 0; i < trials; i++ {
		switch b := <-out; b {
		case 1:
			cnt1++
		case 2:
			cnt2++
		default:
			t.Fatalf("unexpected value %d on channel", b)
		}
	}
	// If the select in the goroutine is fair,
	// cnt1 and cnt2 should be about the same value.
	// See if we're more than 10 sigma away from the expected value.
	// 10 sigma is a lot, but we're ok with some systematic bias as
	// long as it isn't too severe.
	const mean = trials * 0.5
	const variance = trials * 0.5 * (1 - 0.5)
	stddev := math.Sqrt(variance)
	if math.Abs(float64(cnt1-mean)) > 10*stddev {
		t.Errorf("unfair select: in %d trials, results were %d, %d", trials, cnt1, cnt2)
	}
	close(done)
	wg.Wait()
}

func TestChanSendInterface(t *testing.T) {
	type mt struct{}
	m := &mt{}
	c := make(chan any, 1)
	c <- m
	select {
	case c <- m:
	default:
	}
	select {
	case c <- m:
	case c <- &mt{}:
	default:
	}
}

func TestPseudoRandomSend(t *testing.T) {
	n := 100
	for _, chanCap := range []int{0, n} {
		c := make(chan int, chanCap)
		l := make([]int, n)
		var m sync.Mutex
		m.Lock()
		go func() {
			for i := 0; i < n; i++ {
				runtime.Gosched()
				l[i] = <-c
			}
			m.Unlock()
		}()
		for i := 0; i < n; i++ {
			select {
			case c <- 1:
			case c <- 0:
			}
		}
		m.Lock() // wait
		n0 := 0
		n1 := 0
		for _, i := range l {
			n0 += (i + 1) % 2
			n1 += i
		}
		if n0 <= n/10 || n1 <= n/10 {
			t.Errorf("Want pseudorandom, got %d zeros and %d ones (chan cap %d)", n0, n1, chanCap)
		}
	}
}

func TestMultiConsumer(t *testing.T) {
	const nwork = 23
	const niter = 271828

	pn := []int{2, 3, 7, 11, 13, 17, 19, 23, 27, 31}

	q := make(chan int, nwork*3)
	r := make(chan int, nwork*3)

	// workers
	var wg sync.WaitGroup
	for i := 0; i < nwork; i++ {
		wg.Add(1)
		go func(w int) {
			for v := range q {
				// mess with the fifo-ish nature of range
				if pn[w%len(pn)] == v {
					runtime.Gosched()
				}
				r <- v
			}
			wg.Done()
		}(i)
	}

	// feeder & closer
	expect := 0
	go func() {
		for i := 0; i < niter; i++ {
			v := pn[i%len(pn)]
			expect += v
			q <- v
		}
		close(q)  // no more work
		wg.Wait() // workers done
		close(r)  // ... so there can be no more results
	}()

	// consume & check
	n := 0
	s := 0
	for v := range r {
		n++
		s += v
	}
	if n != niter || s != expect {
		t.Errorf("Expected sum %d (got %d) from %d iter (saw %d)",
			expect, s, niter, n)
	}
}

func TestShrinkStackDuringBlockedSend(t *testing.T) {
	// make sure that channel operations still work when we are
	// blocked on a channel send and we shrink the stack.
	// NOTE: this test probably won't fail unless stack1.go:stackDebug
	// is set to >= 1.
	const n = 10
	c := make(chan int)
	done := make(chan struct{})

	go func() {
		for i := 0; i < n; i++ {
			c <- i
			// use lots of stack, briefly.
			stackGrowthRecursive(20)
		}
		done <- struct{}{}
	}()

	for i := 0; i < n; i++ {
		x := <-c
		if x != i {
			t.Errorf("bad channel read: want %d, got %d", i, x)
		}
		// Waste some time so sender can finish using lots of stack
		// and block in channel send.
		time.Sleep(1 * time.Millisecond)
		// trigger GC which will shrink the stack of the sender.
		runtime.GC()
	}
	<-done
}

func TestNoShrinkStackWhileParking(t *testing.T) {
	if runtime.GOOS == "netbsd" && runtime.GOARCH == "arm64" {
		testenv.SkipFlaky(t, 49382)
	}
	if runtime.GOOS == "openbsd" {
		testenv.SkipFlaky(t, 51482)
	}

	// The goal of this test is to trigger a "racy sudog adjustment"
	// throw. Basically, there's a window between when a goroutine
	// becomes available for preemption for stack scanning (and thus,
	// stack shrinking) but before the goroutine has fully parked on a
	// channel. See issue 40641 for more details on the problem.
	//
	// The way we try to induce this failure is to set up two
	// goroutines: a sender and a receiver that communicate across
	// a channel. We try to set up a situation where the sender
	// grows its stack temporarily then *fully* blocks on a channel
	// often. Meanwhile a GC is triggered so that we try to get a
	// mark worker to shrink the sender's stack and race with the
	// sender parking.
	//
	// Unfortunately the race window here is so small that we
	// either need a ridiculous number of iterations, or we add
	// "usleep(1000)" to park_m, just before the unlockf call.
	const n = 10
	send := func(c chan<- int, done chan struct{}) {
		for i := 0; i < n; i++ {
			c <- i
			// Use lots of stack briefly so that
			// the GC is going to want to shrink us
			// when it scans us. Make sure not to
			// do any function calls otherwise
			// in order to avoid us shrinking ourselves
			// when we're preempted.
			stackGrowthRecursive(20)
		}
		done <- struct{}{}
	}
	recv := func(c <-chan int, done chan struct{}) {
		for i := 0; i < n; i++ {
			// Sleep here so that the sender always
			// fully blocks.
			time.Sleep(10 * time.Microsecond)
			<-c
		}
		done <- struct{}{}
	}
	for i := 0; i < n*20; i++ {
		c := make(chan int)
		done := make(chan struct{})
		go recv(c, done)
		go send(c, done)
		// Wait a little bit before triggering
		// the GC to make sure the sender and
		// receiver have gotten into their groove.
		time.Sleep(50 * time.Microsecond)
		runtime.GC()
		<-done
		<-done
	}
}

func TestSelectDuplicateChannel(t *testing.T) {
	// This test makes sure we can queue a G on
	// the same channel multiple times.
	c := make(chan int)
	d := make(chan int)
	e := make(chan int)

	// goroutine A
	go func() {
		select {
		case <-c:
		case <-c:
		case <-d:
		}
		e <- 9
	}()
	time.Sleep(time.Millisecond) // make sure goroutine A gets queued first on c

	// goroutine B
	go func() {
		<-c
	}()
	time.Sleep(time.Millisecond) // make sure goroutine B gets queued on c before continuing

	d <- 7 // wake up A, it dequeues itself from c.  This operation used to corrupt c.recvq.
	<-e    // A tells us it's done
	c <- 8 // wake up B.  This operation used to fail because c.recvq was corrupted (it tries to wake up an already running G instead of B)
}

func TestSelectStackAdjust(t *testing.T) {
	// Test that channel receive slots that contain local stack
	// pointers are adjusted correctly by stack shrinking.
	c := make(chan *int)
	d := make(chan *int)
	ready1 := make(chan bool)
	ready2 := make(chan bool)

	f := func(ready chan bool, dup bool) {
		// Temporarily grow the stack to 10K.
		stackGrowthRecursive((10 << 10) / (128 * 8))

		// We're ready to trigger GC and stack shrink.
		ready <- true

		val := 42
		var cx *int
		cx = &val

		var c2 chan *int
		var d2 chan *int
		if dup {
			c2 = c
			d2 = d
		}

		// Receive from d. cx won't be affected.
		select {
		case cx = <-c:
		case <-c2:
		case <-d:
		case <-d2:
		}

		// Check that pointer in cx was adjusted correctly.
		if cx != &val {
			t.Error("cx no longer points to val")
		} else if val != 42 {
			t.Error("val changed")
		} else {
			*cx = 43
			if val != 43 {
				t.Error("changing *cx failed to change val")
			}
		}
		ready <- true
	}

	go f(ready1, false)
	go f(ready2, true)

	// Let the goroutines get into the select.
	<-ready1
	<-ready2
	time.Sleep(10 * time.Millisecond)

	// Force concurrent GC to shrink the stacks.
	runtime.GC()

	// Wake selects.
	close(d)
	<-ready1
	<-ready2
}

type struct0 struct{}

func BenchmarkMakeChan(b *testing.B) {
	b.Run("Byte", func { b ->
		var x chan byte
		for i := 0; i < b.N; i++ {
			x = make(chan byte, 8)
		}
		close(x)
	})
	b.Run("Int", func { b ->
		var x chan int
		for i := 0; i < b.N; i++ {
			x = make(chan int, 8)
		}
		close(x)
	})
	b.Run("Ptr", func { b ->
		var x chan *byte
		for i := 0; i < b.N; i++ {
			x = make(chan *byte, 8)
		}
		close(x)
	})
	b.Run("Struct", func { b ->
		b.Run("0", func { b ->
			var x chan struct0
			for i := 0; i < b.N; i++ {
				x = make(chan struct0, 8)
			}
			close(x)
		})
		b.Run("32", func { b ->
			var x chan struct32
			for i := 0; i < b.N; i++ {
				x = make(chan struct32, 8)
			}
			close(x)
		})
		b.Run("40", func { b ->
			var x chan struct40
			for i := 0; i < b.N; i++ {
				x = make(chan struct40, 8)
			}
			close(x)
		})
	})
}

func BenchmarkChanNonblocking(b *testing.B) {
	myc := make(chan int)
	b.RunParallel(func { pb -> for pb.Next() {
		select {
		case <-myc:
		default:
		}
	} })
}

func BenchmarkSelectUncontended(b *testing.B) {
	b.RunParallel(func { pb ->
		myc1 := make(chan int, 1)
		myc2 := make(chan int, 1)
		myc1 <- 0
		for pb.Next() {
			select {
			case <-myc1:
				myc2 <- 0
			case <-myc2:
				myc1 <- 0
			}
		}
	})
}

func BenchmarkSelectSyncContended(b *testing.B) {
	myc1 := make(chan int)
	myc2 := make(chan int)
	myc3 := make(chan int)
	done := make(chan int)
	b.RunParallel(func { pb ->
		go func() {
			for {
				select {
				case myc1 <- 0:
				case myc2 <- 0:
				case myc3 <- 0:
				case <-done:
					return
				}
			}
		}()
		for pb.Next() {
			select {
			case <-myc1:
			case <-myc2:
			case <-myc3:
			}
		}
	})
	close(done)
}

func BenchmarkSelectAsyncContended(b *testing.B) {
	procs := runtime.GOMAXPROCS(0)
	myc1 := make(chan int, procs)
	myc2 := make(chan int, procs)
	b.RunParallel(func { pb ->
		myc1 <- 0
		for pb.Next() {
			select {
			case <-myc1:
				myc2 <- 0
			case <-myc2:
				myc1 <- 0
			}
		}
	})
}

func BenchmarkSelectNonblock(b *testing.B) {
	myc1 := make(chan int)
	myc2 := make(chan int)
	myc3 := make(chan int, 1)
	myc4 := make(chan int, 1)
	b.RunParallel(func { pb ->
		for pb.Next() {
			select {
			case <-myc1:
			default:
			}
			select {
			case myc2 <- 0:
			default:
			}
			select {
			case <-myc3:
			default:
			}
			select {
			case myc4 <- 0:
			default:
			}
		}
	})
}

func BenchmarkChanUncontended(b *testing.B) {
	const C = 100
	b.RunParallel(func { pb ->
		myc := make(chan int, C)
		for pb.Next() {
			for i := 0; i < C; i++ {
				myc <- 0
			}
			for i := 0; i < C; i++ {
				<-myc
			}
		}
	})
}

func BenchmarkChanContended(b *testing.B) {
	const C = 100
	myc := make(chan int, C*runtime.GOMAXPROCS(0))
	b.RunParallel(func { pb ->
		for pb.Next() {
			for i := 0; i < C; i++ {
				myc <- 0
			}
			for i := 0; i < C; i++ {
				<-myc
			}
		}
	})
}

func benchmarkChanSync(b *testing.B, work int) {
	const CallsPerSched = 1000
	procs := 2
	N := int32(b.N / CallsPerSched / procs * procs)
	c := make(chan bool, procs)
	myc := make(chan int)
	for p := 0; p < procs; p++ {
		go func() {
			for {
				i := atomic.AddInt32(&N, -1)
				if i < 0 {
					break
				}
				for g := 0; g < CallsPerSched; g++ {
					if i%2 == 0 {
						<-myc
						localWork(work)
						myc <- 0
						localWork(work)
					} else {
						myc <- 0
						localWork(work)
						<-myc
						localWork(work)
					}
				}
			}
			c <- true
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
}

func BenchmarkChanSync(b *testing.B) {
	benchmarkChanSync(b, 0)
}

func BenchmarkChanSyncWork(b *testing.B) {
	benchmarkChanSync(b, 1000)
}

func benchmarkChanProdCons(b *testing.B, chanSize, localWork int) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, 2*procs)
	myc := make(chan int, chanSize)
	for p := 0; p < procs; p++ {
		go func() {
			foo := 0
			for atomic.AddInt32(&N, -1) >= 0 {
				for g := 0; g < CallsPerSched; g++ {
					for i := 0; i < localWork; i++ {
						foo *= 2
						foo /= 2
					}
					myc <- 1
				}
			}
			myc <- 0
			c <- foo == 42
		}()
		go func() {
			foo := 0
			for {
				v := <-myc
				if v == 0 {
					break
				}
				for i := 0; i < localWork; i++ {
					foo *= 2
					foo /= 2
				}
			}
			c <- foo == 42
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
		<-c
	}
}

func BenchmarkChanProdCons0(b *testing.B) {
	benchmarkChanProdCons(b, 0, 0)
}

func BenchmarkChanProdCons10(b *testing.B) {
	benchmarkChanProdCons(b, 10, 0)
}

func BenchmarkChanProdCons100(b *testing.B) {
	benchmarkChanProdCons(b, 100, 0)
}

func BenchmarkChanProdConsWork0(b *testing.B) {
	benchmarkChanProdCons(b, 0, 100)
}

func BenchmarkChanProdConsWork10(b *testing.B) {
	benchmarkChanProdCons(b, 10, 100)
}

func BenchmarkChanProdConsWork100(b *testing.B) {
	benchmarkChanProdCons(b, 100, 100)
}

func BenchmarkSelectProdCons(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, 2*procs)
	myc := make(chan int, 128)
	myclose := make(chan bool)
	for p := 0; p < procs; p++ {
		go func() {
			// Producer: sends to myc.
			foo := 0
			// Intended to not fire during benchmarking.
			mytimer := time.After(time.Hour)
			for atomic.AddInt32(&N, -1) >= 0 {
				for g := 0; g < CallsPerSched; g++ {
					// Model some local work.
					for i := 0; i < 100; i++ {
						foo *= 2
						foo /= 2
					}
					select {
					case myc <- 1:
					case <-mytimer:
					case <-myclose:
					}
				}
			}
			myc <- 0
			c <- foo == 42
		}()
		go func() {
			// Consumer: receives from myc.
			foo := 0
			// Intended to not fire during benchmarking.
			mytimer := time.After(time.Hour)
		loop:
			for {
				select {
				case v := <-myc:
					if v == 0 {
						break loop
					}
				case <-mytimer:
				case <-myclose:
				}
				// Model some local work.
				for i := 0; i < 100; i++ {
					foo *= 2
					foo /= 2
				}
			}
			c <- foo == 42
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
		<-c
	}
}

func BenchmarkReceiveDataFromClosedChan(b *testing.B) {
	count := b.N
	ch := make(chan struct{}, count)
	for i := 0; i < count; i++ {
		ch <- struct{}{}
	}
	close(ch)

	b.ResetTimer()
	for range ch {
	}
}

func BenchmarkChanCreation(b *testing.B) {
	b.RunParallel(func { pb -> for pb.Next() {
		myc := make(chan int, 1)
		myc <- 0
		<-myc
	} })
}

func BenchmarkChanSem(b *testing.B) {
	type Empty struct{}
	myc := make(chan Empty, runtime.GOMAXPROCS(0))
	b.RunParallel(func { pb -> for pb.Next() {
		myc <- Empty{}
		<-myc
	} })
}

func BenchmarkChanPopular(b *testing.B) {
	const n = 1000
	c := make(chan bool)
	var a []chan bool
	var wg sync.WaitGroup
	wg.Add(n)
	for j := 0; j < n; j++ {
		d := make(chan bool)
		a = append(a, d)
		go func() {
			for i := 0; i < b.N; i++ {
				select {
				case <-c:
				case <-d:
				}
			}
			wg.Done()
		}()
	}
	for i := 0; i < b.N; i++ {
		for _, d := range a {
			d <- true
		}
	}
	wg.Wait()
}

func BenchmarkChanClosed(b *testing.B) {
	c := make(chan struct{})
	close(c)
	b.RunParallel(func { pb -> for pb.Next() {
		select {
		case <-c:
		default:
			b.Error("Unreachable")
		}
	} })
}

var (
	alwaysFalse = false
	workSink    = 0
)

func localWork(w int) {
	foo := 0
	for i := 0; i < w; i++ {
		foo /= (foo + 1)
	}
	if alwaysFalse {
		workSink += foo
	}
}
