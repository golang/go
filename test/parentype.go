// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that types can be parenthesized.

package main

func f(interface{})
func g() {}
func main() {
	f(map[string]string{"a":"b","c":"d"})
	f([...]int{1,2,3})
	f(map[string]func(){"a":g,"c":g})
	f(make(chan(<-chan int)))
	f(make(chan<-(chan int)))
}
