// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strconv"
	"strings"
)

type Stringer interface {
	String() string
}

// StringableList is a slice of some type, where the type
// must have a String method.
type StringableList[T Stringer] []T

func (s StringableList[T]) String() string {
	var sb strings.Builder
	for i, v := range s {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(v.String())
	}
	return sb.String()
}

type myint int

func (a myint) String() string {
	return strconv.Itoa(int(a))
}

func main() {
	v := StringableList[myint]{myint(1), myint(2)}

	if got, want := v.String(), "1, 2"; got != want {
		panic(fmt.Sprintf("got %s, want %s", got, want))
	}
}
