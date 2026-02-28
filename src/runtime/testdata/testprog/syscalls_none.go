// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !linux
// +build !linux

package main

func gettid() int {
	return 0
}

func tidExists(tid int) (exists, supported bool) {
	return false, false
}

func getcwd() (string, error) {
	return "", nil
}

func unshareFs() error {
	return nil
}

func chdir(path string) error {
	return nil
}
