// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ogle

import (
	"os"
	"runtime"
)

// An aborter aborts the thread's current computation, usually
// passing the error to a waiting thread.
type aborter interface {
	Abort(err os.Error)
}

type ogleAborter chan os.Error

func (a ogleAborter) Abort(err os.Error) {
	a <- err
	runtime.Goexit()
}

// try executes a computation; if the computation Aborts, try returns
// the error passed to abort.
func try(f func(a aborter)) os.Error {
	a := make(ogleAborter)
	go func() {
		f(a)
		a <- nil
	}()
	err := <-a
	return err
}
