// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

var tests = []interface{}{
	func(x int, s int) int {
		return x << s
	},
	func(x int, s int64) int {
		return x << s
	},
	func(x int, s int32) int {
		return x << s
	},
	func(x int, s int16) int {
		return x << s
	},
	func(x int, s int8) int {
		return x << s
	},
	func(x int, s int) int {
		return x >> s
	},
	func(x int, s int64) int {
		return x >> s
	},
	func(x int, s int32) int {
		return x >> s
	},
	func(x int, s int16) int {
		return x >> s
	},
	func(x int, s int8) int {
		return x >> s
	},
	func(x uint, s int) uint {
		return x << s
	},
	func(x uint, s int64) uint {
		return x << s
	},
	func(x uint, s int32) uint {
		return x << s
	},
	func(x uint, s int16) uint {
		return x << s
	},
	func(x uint, s int8) uint {
		return x << s
	},
	func(x uint, s int) uint {
		return x >> s
	},
	func(x uint, s int64) uint {
		return x >> s
	},
	func(x uint, s int32) uint {
		return x >> s
	},
	func(x uint, s int16) uint {
		return x >> s
	},
	func(x uint, s int8) uint {
		return x >> s
	},
}

func main() {
	for _, t := range tests {
		runTest(reflect.ValueOf(t))
	}
}

func runTest(f reflect.Value) {
	xt := f.Type().In(0)
	st := f.Type().In(1)

	for _, x := range []int{1, 0, -1} {
		for _, s := range []int{-99, -64, -63, -32, -31, -16, -15, -8, -7, -1, 0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 99} {
			args := []reflect.Value{
				reflect.ValueOf(x).Convert(xt),
				reflect.ValueOf(s).Convert(st),
			}
			if s < 0 {
				shouldPanic(func() {
					f.Call(args)
				})
			} else {
				f.Call(args) // should not panic
			}
		}
	}
}

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("did not panic")
		}
	}()
	f()
}
