// [ $A != 6 ]  || errchk $G -e $D/$F.go

// NOTE: This test is not run by 'run.go' and so not run by all.bash.
// To run this test you must use the ./run shell script.

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 2444

package main
func main() {  // ERROR "stack frame too large"
	var arr [1000200030]int32
	arr_bkup := arr
	_ = arr_bkup
}

