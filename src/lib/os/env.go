// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Environment variables.
// Setenv doesn't exist yet: don't have the run-time hooks yet

package os

import os "os"

var (
	ENOENV = NewError("no such environment variable");
)

func Getenv(s string) (v string, err *Error) {
	n := len(s);
	if n == 0 {
		return "", EINVAL
	}
	for i, e := range sys.Envs {
		if len(e) > n && e[n] == '=' && e[0:n] == s {
			return e[n+1:len(e)], nil
		}
	}
	return "", ENOENV
}
