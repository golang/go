// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure return values are always scanned, when
// calling methods (+functions, TODO) with reflect.

package main

import (
	"reflect"
	"runtime/debug"
	"sync"
)

func main() {
	debug.SetGCPercent(1) // run GC frequently
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 2000; i++ {
				_test()
			}
		}()
	}
	wg.Wait()
}

type Stt struct {
	Data interface{}
}

type My struct {
	b byte
}

func (this *My) Run(rawData []byte) (Stt, error) {
	var data string = "hello"
	stt := Stt{
		Data: data,
	}
	return stt, nil
}

func _test() (interface{}, error) {
	f := reflect.ValueOf(&My{}).MethodByName("Run")
	if method, ok := f.Interface().(func([]byte) (Stt, error)); ok {
		s, e := method(nil)
		// The bug in issue27695 happens here, during the return
		// from the above call (at the end of reflect.callMethod
		// when preparing to return). The result value that
		// is assigned to s was not being scanned if GC happens
		// to occur there.
		i := interface{}(s)
		return i, e
	}
	return nil, nil
}
