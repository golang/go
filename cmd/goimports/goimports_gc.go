// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc
// +build gc

package main

import (
	"flag"
	"runtime/trace"
)

var traceProfile = flag.String("trace", "", "trace profile output")

func doTrace() func() {
	if *traceProfile != "" {
		bw, flush := bufferedFileWriter(*traceProfile)
		trace.Start(bw)
		return func() {
			flush()
			trace.Stop()
		}
	}
	return func() {}
}
