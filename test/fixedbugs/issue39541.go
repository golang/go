// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

func sub(args []reflect.Value) []reflect.Value {
	type A struct {
		s int
		t int
	}
	return []reflect.Value{reflect.ValueOf(A{1, 2})}
}

func main() {
	f := reflect.MakeFunc(reflect.TypeOf((func() interface{})(nil)), sub).Interface().(func() interface{})
	c := make(chan bool, 100)
	for i := 0; i < 100; i++ {
		go func() {
			for j := 0; j < 10000; j++ {
				f()
			}
			c <- true
		}()
	}
	for i := 0; i < 100; i++ {
		<-c
	}
}
