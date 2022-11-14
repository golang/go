// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.5
// +build !go1.5

package plan9

func fixwd() {
}

func Getwd() (wd string, err error) {
	fd, err := open(".", O_RDONLY)
	if err != nil {
		return "", err
	}
	defer Close(fd)
	return Fd2path(fd)
}

func Chdir(path string) error {
	return chdir(path)
}
