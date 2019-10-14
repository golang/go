// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !linux

package poll_test

import (
	"errors"
	"os"
	"runtime"
)

func badStateFile() (*os.File, error) {
	return nil, errors.New("not supported on " + runtime.GOOS)
}

func isBadStateFileError(err error) (string, bool) {
	return "", false
}
