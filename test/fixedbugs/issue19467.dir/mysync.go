// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mysync

import "runtime"

type WaitGroup struct {
	Callers []uintptr
}

func (wg *WaitGroup) Add(x int) {
	wg.Callers = make([]uintptr, 32)
	n := runtime.Callers(1, wg.Callers)
	wg.Callers = wg.Callers[:n]
}

func (wg *WaitGroup) Done() {
	wg.Add(-1)
}
