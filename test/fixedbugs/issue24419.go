// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"strings"
)

func growstack(n int) {
	if n > 0 {
		growstack(n - 1)
	}
}

func main() {
	c := make(chan struct{})
	go compare(c)
	go equal(c)
	go indexByte(c)
	go indexByteString(c)
	<-c
	<-c
	<-c
	<-c
}

func compare(c chan struct{}) {
	defer bytes.Compare(nil, nil)
	growstack(10000)
	c <- struct{}{}
}
func equal(c chan struct{}) {
	defer bytes.Equal(nil, nil)
	growstack(10000)
	c <- struct{}{}
}
func indexByte(c chan struct{}) {
	defer bytes.IndexByte(nil, 0)
	growstack(10000)
	c <- struct{}{}
}
func indexByteString(c chan struct{}) {
	defer strings.IndexByte("", 0)
	growstack(10000)
	c <- struct{}{}
}
