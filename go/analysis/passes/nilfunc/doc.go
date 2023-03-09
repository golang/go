// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package nilfunc defines an Analyzer that checks for useless
// comparisons against nil.
//
// # Analyzer nilfunc
//
// nilfunc: check for useless comparisons between functions and nil
//
// A useless comparison is one like f == nil as opposed to f() == nil.
package nilfunc
