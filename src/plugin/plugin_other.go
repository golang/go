// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin,cgo freebsd,cgo

package plugin

import (
	"os"
)

func openFile(f *os.File) (*Plugin, error) {
	return open(f.Name())
}