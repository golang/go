// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

var initCwd, initCwdErr = Getwd()

func executable() (string, error) {
	path, err := syscall.Getexecname()
	if err != nil {
		return path, err
	}
	if len(path) > 0 && path[0] != '/' {
		if initCwdErr != nil {
			return path, initCwdErr
		}
		if len(path) > 2 && path[0:2] == "./" {
			// skip "./"
			path = path[2:]
		}
		return initCwd + "/" + path, nil
	}
	return path, nil
}
