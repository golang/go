// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.7
// +build !go1.7

// TODO(mdempsky): Remove after #44505 is resolved

package noder

import "runtime"

func walkFrames(pcs []uintptr, visit frameVisitor) {
	for _, pc := range pcs {
		fn := runtime.FuncForPC(pc)
		file, line := fn.FileLine(pc)

		visit(file, line, fn.Name(), pc-fn.Entry())
	}
}
