// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/9935
 * Buggy version: 4df302cc3f03328395dc3fefbfba58b7718e4f2f
 * fix commit-id: ed6a100ba38dd51b0888b9a3d3ac6bdbb26c528c
 * Flaky: 100/100
 * Description: This leak is caused by acquiring l.mu.Lock() twice. The fix is
 * to release l.mu.Lock() before acquiring l.mu.Lock for the second time.
 */
package main

import (
	"errors"
	"math/rand"
	"os"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Cockroach9935", Cockroach9935)
}

type loggingT_cockroach9935 struct {
	mu sync.Mutex
}

func (l *loggingT_cockroach9935) outputLogEntry() {
	l.mu.Lock()
	if err := l.createFile(); err != nil {
		l.exit(err)
	}
	l.mu.Unlock()
}

func (l *loggingT_cockroach9935) createFile() error {
	if rand.Intn(8)%4 > 0 {
		return errors.New("")
	}
	return nil
}

func (l *loggingT_cockroach9935) exit(err error) {
	l.mu.Lock() // Blocked forever
	defer l.mu.Unlock()
}

// Example of goroutine leak trace:
//
// G1
//----------------------------
// l.outputLogEntry()
// l.mu.Lock()
// l.createFile()
// l.exit()
// l.mu.Lock()
//-----------G1 leaks---------

func Cockroach9935() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() {
			l := &loggingT_cockroach9935{}
			go l.outputLogEntry() // G1
		}()
	}
}
