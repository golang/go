// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

func TestChanSendInterface(t *testing.T) {
	type mt struct{}
	m := &mt{}
	c := make(chan interface{}, 1)
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
	c := make(chan int)
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
		case c <- 0:
		case c <- 1:
		}
	}
	m.Lock() // wait
	n0 := 0
	n1 := 0
	for _, i := range l {
		n0 += (i + 1) % 2
		n1 += i
		if n0 > n/10 && n1 > n/10 {
			return
		}
	}
	t.Errorf("Want pseudo random, got %d zeros and %d ones", n0, n1)
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

func BenchmarkSelectUncontended(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			myc1 := make(chan int, 1)
			myc2 := make(chan int, 1)
			myc1 <- 0
			for atomic.AddInt32(&N, -1) >= 0 {
				for g := 0; g < CallsPerSched; g++ {
					select {
					case <-myc1:
						myc2 <- 0
					case <-myc2:
						myc1 <- 0
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

func BenchmarkSelectContended(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	myc1 := make(chan int, procs)
	myc2 := make(chan int, procs)
	for p := 0; p < procs; p++ {
		myc1 <- 0
		go func() {
			for atomic.AddInt32(&N, -1) >= 0 {
				for g := 0; g < CallsPerSched; g++ {
					select {
					case <-myc1:
						myc2 <- 0
					case <-myc2:
						myc1 <- 0
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

func BenchmarkSelectNonblock(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			myc1 := make(chan int)
			myc2 := make(chan int)
			myc3 := make(chan int, 1)
			myc4 := make(chan int, 1)
			for atomic.AddInt32(&N, -1) >= 0 {
				for g := 0; g < CallsPerSched; g++ {
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
			}
			c <- true
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
}

func BenchmarkChanUncontended(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			myc := make(chan int, CallsPerSched)
			for atomic.AddInt32(&N, -1) >= 0 {
				for g := 0; g < CallsPerSched; g++ {
					myc <- 0
				}
				for g := 0; g < CallsPerSched; g++ {
					<-myc
				}
			}
			c <- true
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
}

func BenchmarkChanContended(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	myc := make(chan int, procs*CallsPerSched)
	for p := 0; p < procs; p++ {
		go func() {
			for atomic.AddInt32(&N, -1) >= 0 {
				for g := 0; g < CallsPerSched; g++ {
					myc <- 0
				}
				for g := 0; g < CallsPerSched; g++ {
					<-myc
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
						myc <- 0
					} else {
						myc <- 0
						<-myc
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

func BenchmarkChanCreation(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			for atomic.AddInt32(&N, -1) >= 0 {
				for g := 0; g < CallsPerSched; g++ {
					myc := make(chan int, 1)
					myc <- 0
					<-myc
				}
			}
			c <- true
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
}
