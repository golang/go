// Copyright 2021 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package a

// A StoppableWaitGroup waits for a collection of goroutines to finish.
type StoppableWaitGroup struct {
	// i is the internal counter which can store tolerate negative values
	// as opposed the golang's library WaitGroup.
	i *int64
}

// NewStoppableWaitGroup returns a new StoppableWaitGroup. When the 'Stop' is
// executed, following 'Add()' calls won't have any effect.
func NewStoppableWaitGroup() *StoppableWaitGroup {
	return &StoppableWaitGroup{
		i: func() *int64 { i := int64(0); return &i }(),
	}
}
