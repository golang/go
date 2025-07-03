// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
	"os"
)

type S struct{ a, b int }

func crashOnErr1(err error) S {
	if err != nil {
		panic(err)
	}
	return S{} // zero value struct
}

func f1() {
	defer func() {
		if recover() == nil {
			fmt.Println("failed to have expected panic")
			os.Exit(1)
		}
	}()
	fmt.Println(crashOnErr1(errors.New("test error")))
}

func crashOnErr2(err error) S {
	if err != nil {
		panic(err)
	}
	return S{1, 2} // not zero value struct
}

func f2() {
	defer func() {
		if recover() == nil {
			fmt.Println("failed to have expected panic")
			os.Exit(1)
		}
	}()
	fmt.Println(crashOnErr2(errors.New("test error")))
}

func main() {
	f1()
	f2()
}
