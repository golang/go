// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"context"
	"runtime"
)

// Equal reports whether two slices are equal: the same length and all
// elements equal. All floating point NaNs are considered equal.
func SliceEqual[Elem comparable](s1, s2 []Elem) bool {
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

// ReadAll reads from c until the channel is closed or the context is
// canceled, returning all the values read.
func ReadAll[Elem any](ctx context.Context, c <-chan Elem) []Elem {
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

// Merge merges two channels into a single channel.
// This will leave a goroutine running until either both channels are closed
// or the context is canceled, at which point the returned channel is closed.
func Merge[Elem any](ctx context.Context, c1, c2 <-chan Elem) <-chan Elem {
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

// Filter calls f on each value read from c. If f returns true the value
// is sent on the returned channel. This will leave a goroutine running
// until c is closed or the context is canceled, at which point the
// returned channel is closed.
func Filter[Elem any](ctx context.Context, c <-chan Elem, f func(Elem) bool) <-chan Elem {
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

// Sink returns a channel that discards all values sent to it.
// This will leave a goroutine running until the context is canceled
// or the returned channel is closed.
func Sink[Elem any](ctx context.Context) chan<- Elem {
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
type Exclusive[Val any] struct {
	c chan Val
}

// MakeExclusive makes an initialized exclusive value.
func MakeExclusive[Val any](initial Val) *Exclusive[Val] {
	r := &Exclusive[Val]{
		c: make(chan Val, 1),
	}
	r.c <- initial
	return r
}

// Acquire acquires the exclusive value for private use.
// It must be released using the Release method.
func (e *Exclusive[Val]) Acquire() Val {
	return <-e.c
}

// TryAcquire attempts to acquire the value. The ok result reports whether
// the value was acquired. If the value is acquired, it must be released
// using the Release method.
func (e *Exclusive[Val]) TryAcquire() (v Val, ok bool) {
	select {
	case r := <-e.c:
		return r, true
	default:
		return v, false
	}
}

// Release updates and releases the value.
// This method panics if the value has not been acquired.
func (e *Exclusive[Val]) Release(v Val) {
	select {
	case e.c <- v:
	default:
		panic("Exclusive Release without Acquire")
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
func Ranger[Elem any]() (*Sender[Elem], *Receiver[Elem]) {
	c := make(chan Elem)
	d := make(chan struct{})
	s := &Sender[Elem]{
		values: c,
		done:   d,
	}
	r := &Receiver[Elem]{
		values: c,
		done:   d,
	}
	runtime.SetFinalizer(r, (*Receiver[Elem]).finalize)
	return s, r
}

// A Sender is used to send values to a Receiver.
type Sender[Elem any] struct {
	values chan<- Elem
	done   <-chan struct{}
}

// Send sends a value to the receiver. It reports whether the value was sent.
// The value will not be sent if the context is closed or the receiver
// is freed.
func (s *Sender[Elem]) Send(ctx context.Context, v Elem) bool {
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
// After Close is called, the Sender may no longer be used.
func (s *Sender[Elem]) Close() {
	close(s.values)
}

// A Receiver receives values from a Sender.
type Receiver[Elem any] struct {
	values <-chan Elem
	done   chan<- struct{}
}

// Next returns the next value from the channel. The bool result indicates
// whether the value is valid.
func (r *Receiver[Elem]) Next(ctx context.Context) (v Elem, ok bool) {
	select {
	case <-ctx.Done():
	case v, ok = <-r.values:
	}
	return v, ok
}

// finalize is a finalizer for the receiver.
func (r *Receiver[Elem]) finalize() {
	close(r.done)
}
