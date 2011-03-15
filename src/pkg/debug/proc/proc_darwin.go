// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proc

import "os"

// Process tracing is not supported on OS X yet.

func Attach(pid int) (Process, os.Error) {
	return nil, os.NewError("debug/proc not implemented on OS X")
}

func StartProcess(argv0 string, argv []string, attr *os.ProcAttr) (Process, os.Error) {
	return Attach(0)
}
