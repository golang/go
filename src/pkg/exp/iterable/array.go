// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterable

// This file implements the Iterable interface on some primitive types.

type ByteArray []byte

func (a ByteArray) Iter() <-chan interface{} {
	ch := make(chan interface{})
	go func() {
		for _, e := range a {
			ch <- e
		}
		close(ch)
	}()
	return ch
}

type IntArray []int

func (a IntArray) Iter() <-chan interface{} {
	ch := make(chan interface{})
	go func() {
		for _, e := range a {
			ch <- e
		}
		close(ch)
	}()
	return ch
}

type FloatArray []float

func (a FloatArray) Iter() <-chan interface{} {
	ch := make(chan interface{})
	go func() {
		for _, e := range a {
			ch <- e
		}
		close(ch)
	}()
	return ch
}

type StringArray []string

func (a StringArray) Iter() <-chan interface{} {
	ch := make(chan interface{})
	go func() {
		for _, e := range a {
			ch <- e
		}
		close(ch)
	}()
	return ch
}
