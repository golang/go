// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func set(m map[interface{}]interface{}, key interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("set failed: %v", r)
		}
	}()
	m[key] = nil
	return nil
}

func del(m map[interface{}]interface{}, key interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("del failed: %v", r)
		}
	}()
	delete(m, key)
	return nil
}

func main() {
	m := make(map[interface{}]interface{})
	set(m, []int{1, 2, 3})
	set(m, "abc") // used to throw
	del(m, []int{1, 2, 3})
	del(m, "abc") // used to throw
}
