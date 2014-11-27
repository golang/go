// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Import "C" shouldn't be imported.

package main

/*
#define HELLO 1
*/
import "C"

import "fmt"

type Cgo uint32

const (
	// MustScanSubDirs indicates that events were coalesced hierarchically.
	MustScanSubDirs Cgo = 1 << iota
)

func main() {
	_ = C.HELLO
	ck(MustScanSubDirs, "MustScanSubDirs")
}

func ck(day Cgo, str string) {
	if fmt.Sprint(day) != str {
		panic("cgo.go: " + str)
	}
}
