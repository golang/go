// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,cgo

package plugin

import (
	"fmt"
	"os"
)

func openFile(f *os.File) (*Plugin, error) {
	name := fmt.Sprintf("/proc/self/fd/%d", f.Fd())
	cPath := make([]byte, len(name)+1)
	copy(cPath, name)

	return openPath(name, name, cPath)
}