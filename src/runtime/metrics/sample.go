// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package metrics

import (
	_ "runtime" // depends on the runtime via a linkname'd function
	"unsafe"
)

// Sample captures a single metric sample.
type Sample struct {
	// Name is the name of the metric sampled.
	//
	// It must correspond to a name in one of the metric descriptions
	// returned by All.
	Name string

	// Value is the value of the metric sample.
	Value Value
}

// Implemented in the runtime.
func runtime_readMetrics(unsafe.Pointer, int, int)

// Read populates each Value field in the given slice of metric samples.
//
// Desired metrics should be present in the slice with the appropriate name.
// The user of this API is encouraged to re-use the same slice between calls for
// efficiency, but is not required to do so.
//
// Note that re-use has some caveats. Notably, Values should not be read or
// manipulated while a Read with that value is outstanding; that is a data race.
// This property includes pointer-typed Values (for example, Float64Histogram)
// whose underlying storage will be reused by Read when possible. To safely use
// such values in a concurrent setting, all data must be deep-copied.
//
// It is safe to execute multiple Read calls concurrently, but their arguments
// must share no underlying memory. When in doubt, create a new []Sample from
// scratch, which is always safe, though may be inefficient.
//
// Sample values with names not appearing in All will have their Value populated
// as KindBad to indicate that the name is unknown.
func Read(m []Sample) {
	runtime_readMetrics(unsafe.Pointer(&m[0]), len(m), cap(m))
}
