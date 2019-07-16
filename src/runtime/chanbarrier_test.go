// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"sync"
	"testing"
)

type response struct {
}

type myError struct {
}

func (myError) Error() string { return "" }

func doRequest(useSelect bool) (*response, error) {
	type async struct {
		resp *response
		err  error
	}
	ch := make(chan *async, 0)
	done := make(chan struct{}, 0)

	if useSelect {
		go func() {
			select {
			case ch <- &async{resp: nil, err: myError{}}:
			case <-done:
			}
		}()
	} else {
		go func() {
			ch <- &async{resp: nil, err: myError{}}
		}()
	}

	r := <-ch
	runtime.Gosched()
	return r.resp, r.err
}

func TestChanSendSelectBarrier(t *testing.T) {
	testChanSendBarrier(true)
}

func TestChanSendBarrier(t *testing.T) {
	testChanSendBarrier(false)
}

func testChanSendBarrier(useSelect bool) {
	var wg sync.WaitGroup
	var globalMu sync.Mutex
	outer := 100
	inner := 100000
	if testing.Short() || runtime.GOARCH == "wasm" {
		outer = 10
		inner = 1000
	}
	for i := 0; i < outer; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var garbage []byte
			for j := 0; j < inner; j++ {
				_, err := doRequest(useSelect)
				_, ok := err.(myError)
				if !ok {
					panic(1)
				}
				garbage = make([]byte, 1<<10)
			}
			globalMu.Lock()
			global = garbage
			globalMu.Unlock()
		}()
	}
	wg.Wait()
}
