// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"syscall"
)

func gettid() int {
	return syscall.Gettid()
}

func tidExists(tid int) (exists, supported bool) {
	stat, err := ioutil.ReadFile(fmt.Sprintf("/proc/self/task/%d/stat", tid))
	if os.IsNotExist(err) {
		return false, true
	}
	// Check if it's a zombie thread.
	state := bytes.Fields(stat)[2]
	return !(len(state) == 1 && state[0] == 'Z'), true
}
