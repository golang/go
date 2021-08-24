// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"a"
	"context"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"
)

func TestReadAll() {
	c := make(chan int)
	go func() {
		c <- 4
		c <- 2
		c <- 5
		close(c)
	}()
	got := a.ReadAll(context.Background(), c)
	want := []int{4, 2, 5}
	if !a.SliceEqual(got, want) {
		panic(fmt.Sprintf("ReadAll returned %v, want %v", got, want))
	}
}

func TestMerge() {
	c1 := make(chan int)
	c2 := make(chan int)
	go func() {
		c1 <- 1
		c1 <- 3
		c1 <- 5
		close(c1)
	}()
	go func() {
		c2 <- 2
		c2 <- 4
		c2 <- 6
		close(c2)
	}()
	ctx := context.Background()
	got := a.ReadAll(ctx, a.Merge(ctx, c1, c2))
	sort.Ints(got)
	want := []int{1, 2, 3, 4, 5, 6}
	if !a.SliceEqual(got, want) {
		panic(fmt.Sprintf("Merge returned %v, want %v", got, want))
	}
}

func TestFilter() {
	c := make(chan int)
	go func() {
		c <- 1
		c <- 2
		c <- 3
		close(c)
	}()
	even := func(i int) bool { return i%2 == 0 }
	ctx := context.Background()
	got := a.ReadAll(ctx, a.Filter(ctx, c, even))
	want := []int{2}
	if !a.SliceEqual(got, want) {
		panic(fmt.Sprintf("Filter returned %v, want %v", got, want))
	}
}

func TestSink() {
	c := a.Sink[int](context.Background())
	after := time.NewTimer(time.Minute)
	defer after.Stop()
	send := func(v int) {
		select {
		case c <- v:
		case <-after.C:
			panic("timed out sending to Sink")
		}
	}
	send(1)
	send(2)
	send(3)
	close(c)
}

func TestExclusive() {
	val := 0
	ex := a.MakeExclusive(&val)

	var wg sync.WaitGroup
	f := func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			p := ex.Acquire()
			(*p)++
			ex.Release(p)
		}
	}

	wg.Add(2)
	go f()
	go f()

	wg.Wait()
	if val != 20 {
		panic(fmt.Sprintf("after Acquire/Release loop got %d, want 20", val))
	}
}

func TestExclusiveTry() {
	s := ""
	ex := a.MakeExclusive(&s)
	p, ok := ex.TryAcquire()
	if !ok {
		panic("TryAcquire failed")
	}
	*p = "a"

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, ok := ex.TryAcquire()
		if ok {
			panic(fmt.Sprintf("TryAcquire succeeded unexpectedly"))
		}
	}()
	wg.Wait()

	ex.Release(p)

	p, ok = ex.TryAcquire()
	if !ok {
		panic(fmt.Sprintf("TryAcquire failed"))
	}
}

func TestRanger() {
	s, r := a.Ranger[int]()

	ctx := context.Background()
	go func() {
		// Receive one value then exit.
		v, ok := r.Next(ctx)
		if !ok {
			panic(fmt.Sprintf("did not receive any values"))
		} else if v != 1 {
			panic(fmt.Sprintf("received %d, want 1", v))
		}
	}()

	c1 := make(chan bool)
	c2 := make(chan bool)
	go func() {
		defer close(c2)
		if !s.Send(ctx, 1) {
			panic(fmt.Sprintf("Send failed unexpectedly"))
		}
		close(c1)
		if s.Send(ctx, 2) {
			panic(fmt.Sprintf("Send succeeded unexpectedly"))
		}
	}()

	<-c1

	// Force a garbage collection to try to get the finalizers to run.
	runtime.GC()

	select {
	case <-c2:
	case <-time.After(time.Minute):
		panic("Ranger Send should have failed, but timed out")
	}
}

func main() {
	TestReadAll()
	TestMerge()
	TestFilter()
	TestSink()
	TestExclusive()
	TestExclusiveTry()
	TestRanger()
}
