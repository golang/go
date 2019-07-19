// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"fmt"
	"strconv"
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
	var barn sync.Map
	var wg sync.WaitGroup
	// Store
	barn.Store("cow", "moo")
	barn.Store("chicken", "cluck")
	barn.Store("gopher", "go")

	wg.Add(1)

	// search goroutine
	// try to load the gopher key of the map
	// exit the coroutine when it's not found
	go func() {
		i := 0
		for {
			// Load
			_, found := barn.Load("gopher")
			if !found {
				fmt.Println("floop the gopher", strconv.Itoa(i))
				wg.Done()
				break
			}
			i++
		}
	}()

	// Range & Delete
	barn.Range(func(key interface{}, value interface{}) bool {
		if key.(string) == "gopher" {
			fmt.Println("found the gopher")
			barn.Delete(key.(string)) // cast the key interface
			return false              // stop iteration
		}
		return true // continue to the next iteration in the range
	})

	// at this point we wait for goroutine to exit
	// using a regular map this application would have a race condition
	// notice that the number of iterations can greatly differ depending on the platform
	wg.Wait()

	// Output:
	// found the gopher
	// floop the gopher {number of iterations on the search goroutine}
}
