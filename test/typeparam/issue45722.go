// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
)

func try[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func handle(handle func(error)) {
	if issue := recover(); issue != nil {
		if e, ok := issue.(error); ok && e != nil {
			handle(e)
		} else {
			handle(fmt.Errorf("%v", e))
		}
	}
}

func main() {
	defer handle(func(e error) { log.Fatalln(e) })
	_ = try(fmt.Print(""))
}
