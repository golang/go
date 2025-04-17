// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package context_test

import (
	"context"
	"sync"
	"testing"
	"time"
)

// afterFuncContext is a context that's not one of the types
// defined in context.go, that supports registering AfterFuncs.
type afterFuncContext struct {
	mu         sync.Mutex
	afterFuncs map[*byte]func()
	done       chan struct{}
	err        error
}

var _ context.Context = (*afterFuncContext)(nil)

func (c *afterFuncContext) Deadline() (time.Time, bool) {
	return time.Time{}, false
}

func (c *afterFuncContext) Done() <-chan struct{} {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.done == nil {
		c.done = make(chan struct{})
	}
	return c.done
}

func (c *afterFuncContext) Err() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.err
}

func (c *afterFuncContext) Value(key any) any {
	return nil
}

func (c *afterFuncContext) AfterFunc(f func()) func() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	k := new(byte)
	if c.afterFuncs == nil {
		c.afterFuncs = make(map[*byte]func())
	}
	c.afterFuncs[k] = f
	return func() bool {
		c.mu.Lock()
		defer c.mu.Unlock()
		_, ok := c.afterFuncs[k]
		delete(c.afterFuncs, k)
		return ok
	}
}

func (c *afterFuncContext) cancel(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.err != nil {
		return
	}
	c.err = err
	for _, f := range c.afterFuncs {
		go f()
	}
	c.afterFuncs = nil
}

func TestCustomContextAfterFuncCancel(t *testing.T) {
	ctx0 := &afterFuncContext{}
	ctx1, cancel := context.WithCancel(ctx0)
	defer cancel()
	ctx0.cancel(context.Canceled)
	<-ctx1.Done()
}

func TestCustomContextAfterFuncTimeout(t *testing.T) {
	ctx0 := &afterFuncContext{}
	ctx1, cancel := context.WithTimeout(ctx0, veryLongDuration)
	defer cancel()
	ctx0.cancel(context.Canceled)
	<-ctx1.Done()
}

func TestCustomContextAfterFuncAfterFunc(t *testing.T) {
	ctx0 := &afterFuncContext{}
	donec := make(chan struct{})
	stop := context.AfterFunc(ctx0, func() {
		close(donec)
	})
	defer stop()
	ctx0.cancel(context.Canceled)
	<-donec
}

func TestCustomContextAfterFuncUnregisterCancel(t *testing.T) {
	ctx0 := &afterFuncContext{}
	_, cancel1 := context.WithCancel(ctx0)
	_, cancel2 := context.WithCancel(ctx0)
	if got, want := len(ctx0.afterFuncs), 2; got != want {
		t.Errorf("after WithCancel(ctx0): ctx0 has %v afterFuncs, want %v", got, want)
	}
	cancel1()
	cancel2()
	if got, want := len(ctx0.afterFuncs), 0; got != want {
		t.Errorf("after canceling WithCancel(ctx0): ctx0 has %v afterFuncs, want %v", got, want)
	}
}

func TestCustomContextAfterFuncUnregisterTimeout(t *testing.T) {
	ctx0 := &afterFuncContext{}
	_, cancel := context.WithTimeout(ctx0, veryLongDuration)
	if got, want := len(ctx0.afterFuncs), 1; got != want {
		t.Errorf("after WithTimeout(ctx0, d): ctx0 has %v afterFuncs, want %v", got, want)
	}
	cancel()
	if got, want := len(ctx0.afterFuncs), 0; got != want {
		t.Errorf("after canceling WithTimeout(ctx0, d): ctx0 has %v afterFuncs, want %v", got, want)
	}
}

func TestCustomContextAfterFuncUnregisterAfterFunc(t *testing.T) {
	ctx0 := &afterFuncContext{}
	stop := context.AfterFunc(ctx0, func() {})
	if got, want := len(ctx0.afterFuncs), 1; got != want {
		t.Errorf("after AfterFunc(ctx0, f): ctx0 has %v afterFuncs, want %v", got, want)
	}
	stop()
	if got, want := len(ctx0.afterFuncs), 0; got != want {
		t.Errorf("after stopping AfterFunc(ctx0, f): ctx0 has %v afterFuncs, want %v", got, want)
	}
}
