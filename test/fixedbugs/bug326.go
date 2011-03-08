// errchk $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "os"

func f() (_ int, err os.Error) {
	return
}

func g() (x int, _ os.Error) {
	return
}

func h() (_ int, _ os.Error) {
	return
}

func i() (int, os.Error) {
	return	// ERROR "not enough arguments to return"
}

func f1() (_ int, err os.Error) {
	return 1, nil
}

func g1() (x int, _ os.Error) {
	return 1, nil
}

func h1() (_ int, _ os.Error) {
	return 1, nil
}

func ii() (int, os.Error) {
	return 1, nil
}
