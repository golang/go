// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"sync"
)

func main() {
	ready := make(chan struct{})

	var wg sync.WaitGroup
	for _, path := range os.Args[1:] {
		f, err := os.Open(path)
		if err != nil {
			panic(err)
		}

		spawnWait := make(chan struct{})

		wg.Add(1)
		go func(f *os.File) {
			defer f.Close()
			defer wg.Done()

			close(spawnWait)

			<-ready

			var buf [256]byte
			n, err := f.Read(buf[:])
			if err != nil {
				panic(err)
			}
			os.Stderr.Write(buf[:n])
		}(f)

		// Spawn one goroutine at a time.
		<-spawnWait
	}

	println("waiting")
	close(ready)
	wg.Wait()
}
