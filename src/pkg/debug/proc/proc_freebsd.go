// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proc

import "os"

// Process tracing is not supported on FreeBSD yet.

func Attach(pid int) (Process, os.Error) {
	return nil, os.NewError("debug/proc not implemented on FreeBSD")
}

func ForkExec(argv0 string, argv []string, envv []string, dir string, fd []*os.File) (Process, os.Error) {
	return Attach(0)
}
