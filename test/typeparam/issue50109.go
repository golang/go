// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type AnyCacher[T any] interface {
	// Get an item from the cache. Returns the item or nil, and a bool indicating
	// whether the key was found.
	Get(k string) (T, bool)
	// Add an item to the cache, replacing any existing item.
	Set(k string, x T)
}

// Item ...
type Item[T any] struct {
	Object T
}

// AnyCache implements AnyCacher
type AnyCache[T any] struct {
	*anyCache[T]
}

type anyCache[T any] struct {
	items   map[string]Item[T]
	janitor *janitor[T] // Needed for the failure in the issue
}

// Set adds an item to the cache, replacing any existing item.
func (c *anyCache[T]) Set(k string, x T) {
	c.items[k] = Item[T]{
		Object: x,
	}
}

// Get gets an item from the cache. Returns the item or nil, and a bool indicating
// whether the key was found.
func (c *anyCache[T]) Get(k string) (T, bool) {
	// "Inlining" of get and Expired
	item, found := c.items[k]
	if !found {
		var ret T
		return ret, false
	}

	return item.Object, true
}

type janitor[T any] struct {
	stop chan bool
}

func newAnyCache[T any](m map[string]Item[T]) *anyCache[T] {
	c := &anyCache[T]{
		items: m,
	}
	return c
}

// NewAny[T any](...) returns a new AnyCache[T].
func NewAny[T any]() *AnyCache[T] {
	items := make(map[string]Item[T])
	return &AnyCache[T]{newAnyCache(items)}
}

// NewAnyCacher[T any](...) returns an AnyCacher[T] interface.
func NewAnyCacher[T any]() AnyCacher[T] {
	return NewAny[T]()
}

type MyStruct struct {
	Name string
}

func main() {
	// Create a generic cache.
	// All items are cached as interface{} so they need to be cast back to their
	// original type when retrieved.
	// Failure in issue doesn't happen with 'any' replaced by 'interface{}'
	c := NewAnyCacher[any]()

	myStruct := &MyStruct{"MySuperStruct"}

	c.Set("MySuperStruct", myStruct)

	myRawCachedStruct, found := c.Get("MySuperStruct")

	if found {
		// Casting the retrieved object back to its original type
		myCachedStruct := myRawCachedStruct.(*MyStruct)
		fmt.Printf("%s", myCachedStruct.Name)
	} else {
		fmt.Printf("Error: MySuperStruct not found in cache")
	}

	// Output:
	// MySuperStruct
}
