// run -gcflags="-G=3"

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
)

type A[T any] struct {
	F1 string `json:"t1"`
	F2 T      `json:"t2"`
	B  B      `json:"t3"`
}

type B struct {
	F4 int `json:"t4"`
}

func a[T any]() {
	data := `{"t1":"1","t2":2,"t3":{"t4":4}}`
	a1 := A[T]{}
	if err := json.Unmarshal([]byte(data), &a1); err != nil {
		panic(err)
	}
	if bytes, err := json.Marshal(&a1); err != nil {
		panic(err)
	} else if string(bytes) != data {
		panic(string(bytes))
	}
}

func main() {
	a[int]()
}
