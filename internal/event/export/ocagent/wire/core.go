// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wire

// This file contains type that match core proto types

type Timestamp = string

type Int64Value struct {
	Value int64 `json:"value,omitempty"`
}

type DoubleValue struct {
	Value float64 `json:"value,omitempty"`
}
