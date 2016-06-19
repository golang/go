// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go execution tracer.
// The tracer captures a wide range of execution events like goroutine
// creation/blocking/unblocking, syscall enter/exit/block, GC-related events,
// changes of heap size, processor start/stop, etc and writes them to an io.Writer
// in a compact form. A precise nanosecond-precision timestamp and a stack
// trace is captured for most events. A trace can be analyzed later with
// 'go tool trace' command.
package trace

import (
	"io"
	"runtime"
)

// Start enables tracing for the current program.
// While tracing, the trace will be buffered and written to w.
// Start returns an error if tracing is already enabled.
func Start(w io.Writer) error {
	if err := runtime.StartTrace(); err != nil {
		return err
	}
	go func() {
		for {
			data := runtime.ReadTrace()
			if data == nil {
				break
			}
			w.Write(data)
		}
	}()
	return nil
}

// Stop stops the current tracing, if any.
// Stop only returns after all the writes for the trace have completed.
func Stop() {
	runtime.StopTrace()
}
