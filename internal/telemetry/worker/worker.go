// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package worker provides a very simple mechanism to allow telemetry packages
// to work cooperatively and efficiently.
package worker

import (
	"fmt"
	"os"
)

var (
	workQueue = make(chan func(), 1000)
)

func init() {
	go func() {
		for task := range workQueue {
			task()
		}
	}()
}

// Do adds a task to the list of things to work on in the background.
// All tasks will be handled in submission order, and no two tasks will happen
// concurrently so they do not need to do any kind of locking.
// It is safe however to call Do concurrently.
// No promises are made about when the tasks will be performed.
// This function may block, but in general it will return very quickly and
// before the task has been run.
func Do(task func()) {
	select {
	case workQueue <- task:
	default:
		fmt.Fprint(os.Stderr, "work queue is full\n")
		workQueue <- task
	}
}
