// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips140cache

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestCache(t *testing.T) {
	c := new(Cache[key, value])
	checkTrue := func(*value) bool { return true }
	checkFalse := func(*value) bool { return false }
	newNotCalled := func() (*value, error) {
		t.Helper()
		t.Fatal("new called")
		return nil, nil
	}

	k1 := newKey()
	v1 := &value{}

	v, err := c.Get(k1, func() (*value, error) { return v1, nil }, checkTrue)
	expectValue(t, v, err, v1)

	// Cached value is returned if check is true.
	v, err = c.Get(k1, newNotCalled, checkTrue)
	expectValue(t, v, err, v1)

	// New value is returned and cached if check is false.
	v2 := &value{}
	v, err = c.Get(k1, func() (*value, error) { return v2, nil }, checkFalse)
	expectValue(t, v, err, v2)
	v, err = c.Get(k1, newNotCalled, checkTrue)
	expectValue(t, v, err, v2)
	expectMapSize(t, c, 1)

	// Cache is evicted when key becomes unreachable.
	waitUnreachable(t, &k1)
	expectMapSize(t, c, 0)

	// Value is not cached if new returns an error.
	k2 := newKey()
	err1 := errors.New("error")
	_, err = c.Get(k2, func() (*value, error) { return nil, err1 }, checkTrue)
	if err != err1 {
		t.Errorf("got %v, want %v", err, err1)
	}
	expectMapSize(t, c, 0)

	// Value is not replaced if check is false and new returns an error.
	v, err = c.Get(k2, func() (*value, error) { return v1, nil }, checkTrue)
	expectValue(t, v, err, v1)
	_, err = c.Get(k2, func() (*value, error) { return v2, err1 }, checkFalse)
	if err != err1 {
		t.Errorf("got %v, want %v", err, err1)
	}
	v, err = c.Get(k2, newNotCalled, checkTrue)
	expectValue(t, v, err, v1)
	expectMapSize(t, c, 1)

	// Cache is evicted for keys used only once.
	k3 := newKey()
	v, err = c.Get(k3, func() (*value, error) { return v1, nil }, checkTrue)
	expectValue(t, v, err, v1)
	expectMapSize(t, c, 2)
	waitUnreachable(t, &k2)
	waitUnreachable(t, &k3)
	expectMapSize(t, c, 0)

	// When two goroutines race, the returned value may be the new or old one,
	// but the map must shrink to 0.
	keys := make([]*key, 100)
	for i := range keys {
		keys[i] = newKey()
		v1, v2 := &value{}, &value{}
		start := make(chan struct{})
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			<-start
			v, err := c.Get(keys[i], func() (*value, error) { return v1, nil }, checkTrue)
			expectValue(t, v, err, v1, v2)
			wg.Done()
		}()
		go func() {
			<-start
			v, err := c.Get(keys[i], func() (*value, error) { return v2, nil }, checkTrue)
			expectValue(t, v, err, v1, v2)
			wg.Done()
		}()
		close(start)
		wg.Wait()
		v3 := &value{}
		v, err := c.Get(keys[i], func() (*value, error) { return v3, nil }, checkTrue)
		expectValue(t, v, err, v1, v2)
	}
	for i := range keys {
		waitUnreachable(t, &keys[i])
	}
	expectMapSize(t, c, 0)
}

type key struct {
	_ *int
}

type value struct {
	_ *int
}

// newKey allocates a key value on the heap.
//
//go:noinline
func newKey() *key {
	return &key{}
}

func expectValue(t *testing.T, v *value, err error, want ...*value) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range want {
		if v == w {
			return
		}
	}
	t.Errorf("got %p, want %p", v, want)
}

func expectMapSize(t *testing.T, c *Cache[key, value], want int) {
	t.Helper()
	var size int
	// Loop a few times because the AddCleanup might not be done yet.
	for range 10 {
		size = 0
		c.m.Range(func(_, _ any) bool {
			size++
			return true
		})
		if size == want {
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Errorf("got %d, want %d", size, want)
}

func waitUnreachable(t *testing.T, k **key) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	runtime.AddCleanup(*k, func(_ *int) { cancel() }, nil)
	*k = nil
	for ctx.Err() == nil {
		runtime.GC()
	}
	if ctx.Err() != context.Canceled {
		t.Fatal(ctx.Err())
	}
}
