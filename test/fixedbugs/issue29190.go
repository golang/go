// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"strings"
)

type T struct{}

const maxInt = int(^uint(0) >> 1)

func main() {
	s := make([]T, maxInt)
	shouldPanic("cap out of range", func() { s = append(s, T{}) })
	var oneElem = make([]T, 1)
	shouldPanic("cap out of range", func() { s = append(s, oneElem...) })
}

func shouldPanic(str string, f func()) {
	defer func() {
		err := recover()
		if err == nil {
			panic("did not panic")
		}
		s := err.(error).Error()
		if !strings.Contains(s, str) {
			panic("got panic " + s + ", want " + str)
		}
	}()

	f()
}
