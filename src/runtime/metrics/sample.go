// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package metrics

// Sample captures a single metric sample.
type Sample struct {
	// Name is the name of the metric sampled.
	//
	// It must correspond to a name in one of the metric descriptions
	// returned by Descriptions.
	Name string

	// Value is the value of the metric sample.
	Value Value
}

// Read populates each Value field in the given slice of metric samples.
//
// Desired metrics should be present in the slice with the appropriate name.
// The user of this API is encouraged to re-use the same slice between calls.
//
// Metric values with names not appearing in the value returned by Descriptions
// will have the value populated as KindBad to indicate that the name is
// unknown.
func Read(m []Sample) {
	panic("unimplemented")
}
