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
	var barn sync.Map
	var wg sync.WaitGroup
	// Store
	barn.Store("cow", "moo")
	barn.Store("chicken", "cluck")
	barn.Store("gopher", "go")

	wg.Add(1)

	// Scramble the barn coroutines:
	// store and try to not find the pig
	go func() {
		for {
			barn.Store("pig", "mud")
			// Load
			_, found := barn.Load("pig")
			if !found {
				fmt.Println("floop the pig")
				wg.Done()
				break
			}
		}
	}()

	// delete and try to find the pig
	go func() {
		for {
			barn.Delete("pig")
			// Load
			_, found := barn.Load("pig")
			if found {
				fmt.Println("floop the pig")
				wg.Done()
				break
			}
		}
	}()

	// Range & Delete
	barn.Range(func(key interface{}, value interface{}) bool {
		if key.(string) == "chicken" {
			fmt.Println("found the chicken")
			barn.Delete(key.(string)) // cast the key interface
			return false // stop iteration
		}
		return true // continue to the next iteration in the range
	})

	// at this point we wait for the pig to be or not be there
	// using a regular map this application would have a race condition
	// yet using sync map we are safely replicating indetermination
	wg.Wait()

	// Output:
	// found the chicken
	// floop the pig
}
