// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "errors"

var executablePath string // set by ../runtime/os_darwin.go

var initCwd, initCwdErr = Getwd()

func executable() (string, error) {
	ep := executablePath
	if len(ep) == 0 {
		return ep, errors.New("cannot find executable path")
	}
	if ep[0] != '/' {
		if initCwdErr != nil {
			return ep, initCwdErr
		}
		if len(ep) > 2 && ep[0:2] == "./" {
			// skip "./"
			ep = ep[2:]
		}
		ep = initCwd + "/" + ep
	}
	return ep, nil
}
