// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pprof writes runtime profiling data in the format expected
// by the pprof visualization tool.
// For more information about pprof, see
// http://code.google.com/p/google-perftools/.
package pprof

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"runtime"
)

// WriteHeapProfile writes a pprof-formatted heap profile to w.
// If a write to w returns an error, WriteHeapProfile returns that error.
// Otherwise, WriteHeapProfile returns nil.
func WriteHeapProfile(w io.Writer) os.Error {
	// Find out how many records there are (MemProfile(nil, false)),
	// allocate that many records, and get the data.
	// There's a race—more records might be added between
	// the two calls—so allocate a few extra records for safety
	// and also try again if we're very unlucky.
	// The loop should only execute one iteration in the common case.
	var p []runtime.MemProfileRecord
	n, ok := runtime.MemProfile(nil, false)
	for {
		// Allocate room for a slightly bigger profile,
		// in case a few more entries have been added
		// since the call to MemProfile.
		p = make([]runtime.MemProfileRecord, n+50)
		n, ok = runtime.MemProfile(p, false)
		if ok {
			p = p[0:n]
			break
		}
		// Profile grew; try again.
	}

	var total runtime.MemProfileRecord
	for i := range p {
		r := &p[i]
		total.AllocBytes += r.AllocBytes
		total.AllocObjects += r.AllocObjects
		total.FreeBytes += r.FreeBytes
		total.FreeObjects += r.FreeObjects
	}

	// Technically the rate is MemProfileRate not 2*MemProfileRate,
	// but early versions of the C++ heap profiler reported 2*MemProfileRate,
	// so that's what pprof has come to expect.
	b := bufio.NewWriter(w)
	fmt.Fprintf(b, "heap profile: %d: %d [%d: %d] @ heap/%d\n",
		total.InUseObjects(), total.InUseBytes(),
		total.AllocObjects, total.AllocBytes,
		2*runtime.MemProfileRate)

	for i := range p {
		r := &p[i]
		fmt.Fprintf(b, "%d: %d [%d: %d] @",
			r.InUseObjects(), r.InUseBytes(),
			r.AllocObjects, r.AllocBytes)
		for _, pc := range r.Stack() {
			fmt.Fprintf(b, " %#x", pc)
		}
		fmt.Fprintf(b, "\n")
	}
	return b.Flush()
}
