// compile

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to run 6g out of registers.  Issue 2669.

package p

type y struct {
	num int
}

func zzz () {
    k := make([]byte, 10)
	arr := make ([]*y, 0)
    for s := range arr {
        x := make([]byte, 10)
        for i := 0; i < 100 ; i++ {
            x[i] ^= k[i-arr[s].num%3]
        }
    }
}
