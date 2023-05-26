// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package append defines an Analyzer that detects
// if there is only one variable in append.
//
// # Analyzer append
//
// append: check for missing values after append
//
// This checker reports append with no values.
//
// For example:
//
//	sli := []string{"a", "b", "c"}
//	sli = append(sli)
//
//	it would report "append with no values"
package append
