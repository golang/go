// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Crashes 6g, 8g
// http://code.google.com/p/go/issues/detail?id=238

package main

func main() {
	bar := make(chan bool);
	select {
	case _ = <-bar:
		return
	}
}

/*
6g bug218.go 
<epoch>: fatal error: dowidth: unknown type: blank
*/
