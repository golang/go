package main

import (
	"runtime"
	"sync"
	"time"
)

// This is a set of micro-tests with obvious goroutine leaks that
// ensures goroutine leak detection works.

func init() {
	register("NilRecv", NilRecv)
	register("NilSend", NilSend)
	register("SelectNoCases", SelectNoCases)
	register("ChanRecv", ChanRecv)
	register("ChanSend", ChanSend)
	register("Select", Select)
	register("WaitGroup", WaitGroup)
	register("MutexStack", MutexStack)
	register("MutexHeap", MutexHeap)
	register("RWMutexRLock", RWMutexRLock)
	register("RWMutexLock", RWMutexLock)
	register("Cond", Cond)
	register("Mixed", Mixed)
	register("NoLeakGlobal", NoLeakGlobal)
}

func NilRecv() {
	go func() {
		var c chan int
		<-c
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func NilSend() {
	go func() {
		var c chan int
		c <- 0
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func ChanRecv() {
	go func() {
		<-make(chan int)
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func SelectNoCases() {
	go func() {
		select {}
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func ChanSend() {
	go func() {
		make(chan int) <- 0
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func Select() {
	go func() {
		select {
		case make(chan int) <- 0:
		case <-make(chan int):
		}
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func WaitGroup() {
	go func() {
		var wg sync.WaitGroup
		wg.Add(1)
		wg.Wait()
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func MutexStack() {
	for i := 0; i < 1000; i++ {
		go func() {
			var mu sync.Mutex
			mu.Lock()
			mu.Lock()
			panic("should not be reached")
		}()
	}
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
	time.Sleep(10 * time.Millisecond)
}

func MutexHeap() {
	for i := 0; i < 1000; i++ {
		go func() {
			mu := &sync.Mutex{}
			go func() {
				mu.Lock()
				mu.Lock()
				panic("should not be reached")
			}()
		}()
	}
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
	time.Sleep(10 * time.Millisecond)
}

func RWMutexRLock() {
	go func() {
		mu := &sync.RWMutex{}
		mu.Lock()
		mu.RLock()
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func RWMutexLock() {
	go func() {
		mu := &sync.RWMutex{}
		mu.Lock()
		mu.Lock()
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func Cond() {
	go func() {
		cond := sync.NewCond(&sync.Mutex{})
		cond.L.Lock()
		cond.Wait()
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func Mixed() {
	go func() {
		ch := make(chan int)
		wg := sync.WaitGroup{}
		wg.Add(1)
		go func() {
			ch <- 0
			wg.Done()
			panic("should not be reached")
		}()
		wg.Wait()
		<-ch
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

var ch = make(chan int)

// No leak should be reported by this test
func NoLeakGlobal() {
	go func() {
		<-ch
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}
