// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/xml"
	"fmt"
)

type A[T, U any] struct {
	Name T `xml:"name"`
	Data U `xml:"data"`
}

func main() {
	src := &A[string, int]{Name: "name", Data: 1}
	data, err := xml.Marshal(src)
	if err != nil {
		panic(err)
	}
	dst := &A[string, int]{}
	err = xml.Unmarshal(data, dst)
	if err != nil {
		panic(err)
	}
	if *src != *dst {
		panic(fmt.Sprintf("wanted %#v got %#v", src, dst))
	}
}
