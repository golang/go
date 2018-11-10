// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo

func f(x []interface{}) (err error) {
	for _, d := range x {
		_, ok := d.(*int)
		if ok {
			return
		}
	}
	return
}
