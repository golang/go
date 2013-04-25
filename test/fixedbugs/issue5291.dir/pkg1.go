// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkg1

import (
	"runtime"
)

type T2 *[]string

type Data struct {
	T1 *[]T2
}

func CrashCall() (err error) {
	var d Data

	for count := 0; count < 10; count++ {
		runtime.GC()

		len := 2 // crash when >=2
		x := make([]T2, len)

		d = Data{T1: &x}

		for j := 0; j < len; j++ {
			y := make([]string, 1)
			(*d.T1)[j] = &y
		}
	}
	return nil
}
