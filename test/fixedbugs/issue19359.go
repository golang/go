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

func addInt(m map[interface{}]int, key interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("addInt failed: %v", r)
		}
	}()
	m[key] += 2018
	return nil
}

func addStr(m map[interface{}]string, key interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("addStr failed: %v", r)
		}
	}()
	m[key] += "hello, go"
	return nil
}

func appendInt(m map[interface{}][]int, key interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("appendInt failed: %v", r)
		}
	}()
	m[key] = append(m[key], 2018)
	return nil
}

func appendStr(m map[interface{}][]string, key interface{}) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("addStr failed: %v", r)
		}
	}()
	m[key] = append(m[key], "hello, go")
	return nil
}

func main() {
	m := make(map[interface{}]interface{})
	set(m, []int{1, 2, 3})
	set(m, "abc") // used to throw
	del(m, []int{1, 2, 3})
	del(m, "abc") // used to throw

	mi := make(map[interface{}]int)
	addInt(mi, []int{1, 2, 3})
	addInt(mi, "abc") // used to throw

	ms := make(map[interface{}]string)
	addStr(ms, []int{1, 2, 3})
	addStr(ms, "abc") // used to throw

	mia := make(map[interface{}][]int)
	appendInt(mia, []int{1, 2, 3})

	msa := make(map[interface{}][]string)
	appendStr(msa, "abc") // used to throw
}
