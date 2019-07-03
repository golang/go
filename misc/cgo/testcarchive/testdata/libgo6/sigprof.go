// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"io/ioutil"
	"runtime/pprof"
)

import "C"

//export go_start_profile
func go_start_profile() {
	pprof.StartCPUProfile(ioutil.Discard)
}

//export go_stop_profile
func go_stop_profile() {
	pprof.StopCPUProfile()
}

func main() {
}
