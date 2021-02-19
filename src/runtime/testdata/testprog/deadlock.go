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
	register("RecursivePanic2", RecursivePanic2)
	register("RecursivePanic3", RecursivePanic3)
	register("RecursivePanic4", RecursivePanic4)
	register("RecursivePanic5", RecursivePanic5)
	register("GoexitExit", GoexitExit)
	register("GoNil", GoNil)
	register("MainGoroutineID", MainGoroutineID)
	register("Breakpoint", Breakpoint)
	register("GoexitInPanic", GoexitInPanic)
	register("PanicAfterGoexit", PanicAfterGoexit)
	register("RecoveredPanicAfterGoexit", RecoveredPanicAfterGoexit)
	register("RecoverBeforePanicAfterGoexit", RecoverBeforePanicAfterGoexit)
	register("RecoverBeforePanicAfterGoexit2", RecoverBeforePanicAfterGoexit2)
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

// Same as RecursivePanic, but do the first recover and the second panic in
// separate defers, and make sure they are executed in the correct order.
func RecursivePanic2() {
	func() {
		defer func() {
			fmt.Println(recover())
		}()
		var x [8192]byte
		func(x [8192]byte) {
			defer func() {
				panic("second panic")
			}()
			defer func() {
				fmt.Println(recover())
			}()
			panic("first panic")
		}(x)
	}()
	panic("third panic")
}

// Make sure that the first panic finished as a panic, even though the second
// panic was recovered
func RecursivePanic3() {
	defer func() {
		defer func() {
			recover()
		}()
		panic("second panic")
	}()
	panic("first panic")
}

// Test case where a single defer recovers one panic but starts another panic. If
// the second panic is never recovered, then the recovered first panic will still
// appear on the panic stack (labeled '[recovered]') and the runtime stack.
func RecursivePanic4() {
	defer func() {
		recover()
		panic("second panic")
	}()
	panic("first panic")
}

// Test case where we have an open-coded defer higher up the stack (in two), and
// in the current function (three) we recover in a defer while we still have
// another defer to be processed.
func RecursivePanic5() {
	one()
	panic("third panic")
}

//go:noinline
func one() {
	two()
}

//go:noinline
func two() {
	defer func() {
	}()

	three()
}

//go:noinline
func three() {
	defer func() {
	}()

	defer func() {
		fmt.Println(recover())
	}()

	defer func() {
		fmt.Println(recover())
		panic("second panic")
	}()

	panic("first panic")
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

func RecoverBeforePanicAfterGoexit() {
	// 1. defer a function that recovers
	// 2. defer a function that panics
	// 3. call goexit
	// Goexit runs the #2 defer. Its panic
	// is caught by the #1 defer.  For Goexit, we explicitly
	// resume execution in the Goexit loop, instead of resuming
	// execution in the caller (which would make the Goexit disappear!)
	defer func() {
		r := recover()
		if r == nil {
			panic("bad recover")
		}
	}()
	defer func() {
		panic("hello")
	}()
	runtime.Goexit()
}

func RecoverBeforePanicAfterGoexit2() {
	for i := 0; i < 2; i++ {
		defer func() {
		}()
	}
	// 1. defer a function that recovers
	// 2. defer a function that panics
	// 3. call goexit
	// Goexit runs the #2 defer. Its panic
	// is caught by the #1 defer.  For Goexit, we explicitly
	// resume execution in the Goexit loop, instead of resuming
	// execution in the caller (which would make the Goexit disappear!)
	defer func() {
		r := recover()
		if r == nil {
			panic("bad recover")
		}
	}()
	defer func() {
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
