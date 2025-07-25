package main

import (
	"runtime"
	"sync"
	"time"
)

func init() {
	register("GoroutineLeakNilRecv", GoroutineLeakNilRecv)
	register("GoroutineLeakNilSend", GoroutineLeakNilSend)
	register("GoroutineLeakSelectNoCases", GoroutineLeakSelectNoCases)
	register("GoroutineLeakChanRecv", GoroutineLeakChanRecv)
	register("GoroutineLeakChanSend", GoroutineLeakChanSend)
	register("GoroutineLeakSelect", GoroutineLeakSelect)
	register("GoroutineLeakWaitGroup", GoroutineLeakWaitGroup)
	register("GoroutineLeakMutexStack", GoroutineLeakMutexStack)
	register("GoroutineLeakMutexHeap", GoroutineLeakMutexHeap)
	register("GoroutineLeakRWMutexRLock", GoroutineLeakRWMutexRLock)
	register("GoroutineLeakRWMutexLock", GoroutineLeakRWMutexLock)
	register("GoroutineLeakCond", GoroutineLeakCond)
	register("GoroutineLeakMixed", GoroutineLeakMixed)
	register("NoGoroutineLeakGlobal", NoGoroutineLeakGlobal)
}

func GoroutineLeakNilRecv() {
	go func() {
		var c chan int
		<-c
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func GoroutineLeakNilSend() {
	go func() {
		var c chan int
		c <- 0
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func GoroutineLeakChanRecv() {
	go func() {
		<-make(chan int)
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func GoroutineLeakSelectNoCases() {
	go func() {
		select {}
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func GoroutineLeakChanSend() {
	go func() {
		make(chan int) <- 0
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func GoroutineLeakSelect() {
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

func GoroutineLeakWaitGroup() {
	go func() {
		var wg sync.WaitGroup
		wg.Add(1)
		wg.Wait()
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func GoroutineLeakMutexStack() {
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

func GoroutineLeakMutexHeap() {
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

func GoroutineLeakRWMutexRLock() {
	go func() {
		mu := &sync.RWMutex{}
		mu.Lock()
		mu.RLock()
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func GoroutineLeakRWMutexLock() {
	go func() {
		mu := &sync.RWMutex{}
		mu.Lock()
		mu.Lock()
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func GoroutineLeakCond() {
	go func() {
		cond := sync.NewCond(&sync.Mutex{})
		cond.L.Lock()
		cond.Wait()
		panic("should not be reached")
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}

func GoroutineLeakMixed() {
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
func NoGoroutineLeakGlobal() {
	go func() {
		<-ch
	}()
	time.Sleep(10 * time.Millisecond)
	runtime.GC()
}
