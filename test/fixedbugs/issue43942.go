// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "log"

func main() {
	defer func() {
		expect(5, recover())
	}()
	func() {
		expect(nil, recover())
		defer func() {
			defer func() {
				defer func() {
					defer func() {
						expect(3, recover())
					}()
					defer panic(3)
					panic(2)
				}()
				defer func() {
					expect(1, recover())
				}()
				panic(1)
			}()
		}()
	}()
	func() {
		for {
			defer func() {
				defer panic(5)
			}()
			break
		}
		panic(4)
	}()
}

func expect(want, have interface{}) {
	if want != have {
		log.Fatalf("want %v, have %v", want, have)
	}
}
