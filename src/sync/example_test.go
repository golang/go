// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"fmt"
	"sync"
)

type httpPkg struct{}

func (httpPkg) Get(url string) {}

var http httpPkg

// This example fetches several URLs concurrently,
// using a WaitGroup to block until all the fetches are complete.
func ExampleWaitGroup() {
	var wg sync.WaitGroup
	var urls = []string{
		"http://www.golang.org/",
		"http://www.google.com/",
		"http://www.somestupidname.com/",
	}
	for _, url := range urls {
		// Increment the WaitGroup counter.
		wg.Add(1)
		// Launch a goroutine to fetch the URL.
		go func(url string) {
			// Decrement the counter when the goroutine completes.
			defer wg.Done()
			// Fetch the URL.
			http.Get(url)
		}(url)
	}
	// Wait for all HTTP fetches to complete.
	wg.Wait()
}

func ExampleOnce() {
	var once sync.Once
	onceBody := func() {
		fmt.Println("Only once")
	}
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			once.Do(onceBody)
			done <- true
		}()
	}
	for i := 0; i < 10; i++ {
		<-done
	}
	// Output:
	// Only once
}

func ExampleMap() {
	var map sync.Map
	// Store
	map.Store("cow", "moo")
	map.Store("chicken", "cluck")
	// Load
	_, found := map.Load("gopher")
	if !found {
		fmt.Print("no gopher")
	}

	map.Store("gopher", "go")
	gopher, found := map.Load("gopher")
	fmt.Println(gopher, found)

	// Delete
	map.Delete("chicken")
	// Range
	db.Memdb.Range(func(key interface{}, value interface{}) bool {
		fmt.Println(key, value)
	})

	// Output:
	// no gopher
	// go true
	// cow moo
	// gopher go
}