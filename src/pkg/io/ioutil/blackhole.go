// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ioutil

var blackHoleBuf = make(chan []byte, 1)

func blackHole() []byte {
	select {
	case b := <-blackHoleBuf:
		return b
	default:
	}
	return make([]byte, 8192)
}

func blackHolePut(p []byte) {
	select {
	case blackHoleBuf <- p:
	default:
	}
}
