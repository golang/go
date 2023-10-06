// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package appends defines an Analyzer that detects
// if there is only one variable in append.
//
// # Analyzer appends
//
// appends: check for missing values after append
//
// This checker reports calls to append that pass
// no values to be appended to the slice.
//
//	s := []string{"a", "b", "c"}
//	_ = append(s)
//
// Such calls are always no-ops and often indicate an
// underlying mistake.
package appends
