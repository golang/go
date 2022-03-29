// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package chans provides utility functions for working with channels.
package main

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"
)

// _Equal reports whether two slices are equal: the same length and all
// elements equal. All floating point NaNs are considered equal.
func _SliceEqual[Elem comparable](s1, s2 []Elem) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, v1 := range s1 {
		v2 := s2[i]
		if v1 != v2 {
			isNaN := func(f Elem) bool { return f != f }
			if !isNaN(v1) || !isNaN(v2) {
				return false
			}
		}
	}
	return true
}

// _ReadAll reads from c until the channel is closed or the context is
// canceled, returning all the values read.
func _ReadAll[Elem any](ctx context.Context, c <-chan Elem) []Elem {
	var r []Elem
	for {
		select {
		case <-ctx.Done():
			return r
		case v, ok := <-c:
			if !ok {
				return r
			}
			r = append(r, v)
		}
	}
}

// _Merge merges two channels into a single channel.
// This will leave a goroutine running until either both channels are closed
// or the context is canceled, at which point the returned channel is closed.
func _Merge[Elem any](ctx context.Context, c1, c2 <-chan Elem) <-chan Elem {
	r := make(chan Elem)
	go func(ctx context.Context, c1, c2 <-chan Elem, r chan<- Elem) {
		defer close(r)
		for c1 != nil || c2 != nil {
			select {
			case <-ctx.Done():
				return
			case v1, ok := <-c1:
				if ok {
					r <- v1
				} else {
					c1 = nil
				}
			case v2, ok := <-c2:
				if ok {
					r <- v2
				} else {
					c2 = nil
				}
			}
		}
	}(ctx, c1, c2, r)
	return r
}

// _Filter calls f on each value read from c. If f returns true the value
// is sent on the returned channel. This will leave a goroutine running
// until c is closed or the context is canceled, at which point the
// returned channel is closed.
func _Filter[Elem any](ctx context.Context, c <-chan Elem, f func(Elem) bool) <-chan Elem {
	r := make(chan Elem)
	go func(ctx context.Context, c <-chan Elem, f func(Elem) bool, r chan<- Elem) {
		defer close(r)
		for {
			select {
			case <-ctx.Done():
				return
			case v, ok := <-c:
				if !ok {
					return
				}
				if f(v) {
					r <- v
				}
			}
		}
	}(ctx, c, f, r)
	return r
}

// _Sink returns a channel that discards all values sent to it.
// This will leave a goroutine running until the context is canceled
// or the returned channel is closed.
func _Sink[Elem any](ctx context.Context) chan<- Elem {
	r := make(chan Elem)
	go func(ctx context.Context, r <-chan Elem) {
		for {
			select {
			case <-ctx.Done():
				return
			case _, ok := <-r:
				if !ok {
					return
				}
			}
		}
	}(ctx, r)
	return r
}

// An Exclusive is a value that may only be used by a single goroutine
// at a time. This is implemented using channels rather than a mutex.
type _Exclusive[Val any] struct {
	c chan Val
}

// _MakeExclusive makes an initialized exclusive value.
func _MakeExclusive[Val any](initial Val) *_Exclusive[Val] {
	r := &_Exclusive[Val]{
		c: make(chan Val, 1),
	}
	r.c <- initial
	return r
}

// _Acquire acquires the exclusive value for private use.
// It must be released using the Release method.
func (e *_Exclusive[Val]) Acquire() Val {
	return <-e.c
}

// TryAcquire attempts to acquire the value. The ok result reports whether
// the value was acquired. If the value is acquired, it must be released
// using the Release method.
func (e *_Exclusive[Val]) TryAcquire() (v Val, ok bool) {
	select {
	case r := <-e.c:
		return r, true
	default:
		return v, false
	}
}

// Release updates and releases the value.
// This method panics if the value has not been acquired.
func (e *_Exclusive[Val]) Release(v Val) {
	select {
	case e.c <- v:
	default:
		panic("_Exclusive Release without Acquire")
	}
}

// Ranger returns a Sender and a Receiver. The Receiver provides a
// Next method to retrieve values. The Sender provides a Send method
// to send values and a Close method to stop sending values. The Next
// method indicates when the Sender has been closed, and the Send
// method indicates when the Receiver has been freed.
//
// This is a convenient way to exit a goroutine sending values when
// the receiver stops reading them.
func _Ranger[Elem any]() (*_Sender[Elem], *_Receiver[Elem]) {
	c := make(chan Elem)
	d := make(chan struct{})
	s := &_Sender[Elem]{
		values: c,
		done:   d,
	}
	r := &_Receiver[Elem]{
		values: c,
		done:   d,
	}
	runtime.SetFinalizer(r, (*_Receiver[Elem]).finalize)
	return s, r
}

// A _Sender is used to send values to a Receiver.
type _Sender[Elem any] struct {
	values chan<- Elem
	done   <-chan struct{}
}

// Send sends a value to the receiver. It reports whether the value was sent.
// The value will not be sent if the context is closed or the receiver
// is freed.
func (s *_Sender[Elem]) Send(ctx context.Context, v Elem) bool {
	select {
	case <-ctx.Done():
		return false
	case s.values <- v:
		return true
	case <-s.done:
		return false
	}
}

// Close tells the receiver that no more values will arrive.
// After Close is called, the _Sender may no longer be used.
func (s *_Sender[Elem]) Close() {
	close(s.values)
}

// A _Receiver receives values from a _Sender.
type _Receiver[Elem any] struct {
	values <-chan Elem
	done   chan<- struct{}
}

// Next returns the next value from the channel. The bool result indicates
// whether the value is valid.
func (r *_Receiver[Elem]) Next(ctx context.Context) (v Elem, ok bool) {
	select {
	case <-ctx.Done():
	case v, ok = <-r.values:
	}
	return v, ok
}

// finalize is a finalizer for the receiver.
func (r *_Receiver[Elem]) finalize() {
	close(r.done)
}

func TestReadAll() {
	c := make(chan int)
	go func() {
		c <- 4
		c <- 2
		c <- 5
		close(c)
	}()
	got := _ReadAll(context.Background(), c)
	want := []int{4, 2, 5}
	if !_SliceEqual(got, want) {
		panic(fmt.Sprintf("_ReadAll returned %v, want %v", got, want))
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
	got := _ReadAll(ctx, _Merge(ctx, c1, c2))
	sort.Ints(got)
	want := []int{1, 2, 3, 4, 5, 6}
	if !_SliceEqual(got, want) {
		panic(fmt.Sprintf("_Merge returned %v, want %v", got, want))
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
	got := _ReadAll(ctx, _Filter(ctx, c, even))
	want := []int{2}
	if !_SliceEqual(got, want) {
		panic(fmt.Sprintf("_Filter returned %v, want %v", got, want))
	}
}

func TestSink() {
	c := _Sink[int](context.Background())
	after := time.NewTimer(time.Minute)
	defer after.Stop()
	send := func(v int) {
		select {
		case c <- v:
		case <-after.C:
			panic("timed out sending to _Sink")
		}
	}
	send(1)
	send(2)
	send(3)
	close(c)
}

func TestExclusive() {
	val := 0
	ex := _MakeExclusive(&val)

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
	ex := _MakeExclusive(&s)
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
	s, r := _Ranger[int]()

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
		panic("_Ranger Send should have failed, but timed out")
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
