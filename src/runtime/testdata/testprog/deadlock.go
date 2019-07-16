// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"time"
)

func init() {
	registerInit("InitDeadlock", InitDeadlock)
	registerInit("NoHelperGoroutines", NoHelperGoroutines)

	register("SimpleDeadlock", SimpleDeadlock)
	register("LockedDeadlock", LockedDeadlock)
	register("LockedDeadlock2", LockedDeadlock2)
	register("GoexitDeadlock", GoexitDeadlock)
	register("StackOverflow", StackOverflow)
	register("ThreadExhaustion", ThreadExhaustion)
	register("RecursivePanic", RecursivePanic)
	register("GoexitExit", GoexitExit)
	register("GoNil", GoNil)
	register("MainGoroutineID", MainGoroutineID)
	register("Breakpoint", Breakpoint)
	register("GoexitInPanic", GoexitInPanic)
	register("PanicAfterGoexit", PanicAfterGoexit)
	register("RecoveredPanicAfterGoexit", RecoveredPanicAfterGoexit)
	register("PanicTraceback", PanicTraceback)
	register("GoschedInPanic", GoschedInPanic)
	register("SyscallInPanic", SyscallInPanic)
	register("PanicLoop", PanicLoop)
}

func SimpleDeadlock() {
	select {}
	panic("not reached")
}

func InitDeadlock() {
	select {}
	panic("not reached")
}

func LockedDeadlock() {
	runtime.LockOSThread()
	select {}
}

func LockedDeadlock2() {
	go func() {
		runtime.LockOSThread()
		select {}
	}()
	time.Sleep(time.Millisecond)
	select {}
}

func GoexitDeadlock() {
	F := func() {
		for i := 0; i < 10; i++ {
		}
	}

	go F()
	go F()
	runtime.Goexit()
}

func StackOverflow() {
	var f func() byte
	f = func() byte {
		var buf [64 << 10]byte
		return buf[0] + f()
	}
	debug.SetMaxStack(1474560)
	f()
}

func ThreadExhaustion() {
	debug.SetMaxThreads(10)
	c := make(chan int)
	for i := 0; i < 100; i++ {
		go func() {
			runtime.LockOSThread()
			c <- 0
			select {}
		}()
		<-c
	}
}

func RecursivePanic() {
	func() {
		defer func() {
			fmt.Println(recover())
		}()
		var x [8192]byte
		func(x [8192]byte) {
			defer func() {
				if err := recover(); err != nil {
					panic("wrap: " + err.(string))
				}
			}()
			panic("bad")
		}(x)
	}()
	panic("again")
}

func GoexitExit() {
	println("t1")
	go func() {
		time.Sleep(time.Millisecond)
	}()
	i := 0
	println("t2")
	runtime.SetFinalizer(&i, func(p *int) {})
	println("t3")
	runtime.GC()
	println("t4")
	runtime.Goexit()
}

func GoNil() {
	defer func() {
		recover()
	}()
	var f func()
	go f()
	select {}
}

func MainGoroutineID() {
	panic("test")
}

func NoHelperGoroutines() {
	i := 0
	runtime.SetFinalizer(&i, func(p *int) {})
	time.AfterFunc(time.Hour, func() {})
	panic("oops")
}

func Breakpoint() {
	runtime.Breakpoint()
}

func GoexitInPanic() {
	go func() {
		defer func() {
			runtime.Goexit()
		}()
		panic("hello")
	}()
	runtime.Goexit()
}

type errorThatGosched struct{}

func (errorThatGosched) Error() string {
	runtime.Gosched()
	return "errorThatGosched"
}

func GoschedInPanic() {
	panic(errorThatGosched{})
}

type errorThatPrint struct{}

func (errorThatPrint) Error() string {
	fmt.Println("1")
	fmt.Println("2")
	return "3"
}

func SyscallInPanic() {
	panic(errorThatPrint{})
}

func PanicAfterGoexit() {
	defer func() {
		panic("hello")
	}()
	runtime.Goexit()
}

func RecoveredPanicAfterGoexit() {
	defer func() {
		defer func() {
			r := recover()
			if r == nil {
				panic("bad recover")
			}
		}()
		panic("hello")
	}()
	runtime.Goexit()
}

func PanicTraceback() {
	pt1()
}

func pt1() {
	defer func() {
		panic("panic pt1")
	}()
	pt2()
}

func pt2() {
	defer func() {
		panic("panic pt2")
	}()
	panic("hello")
}

type panicError struct{}

func (*panicError) Error() string {
	panic("double error")
}

func PanicLoop() {
	panic(&panicError{})
}
