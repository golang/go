// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package assign defines an Analyzer that detects useless assignments.
//
// # Analyzer assign
//
// assign: check for useless assignments
//
// This checker reports assignments of the form x = x or a[i] = a[i].
// These are almost always useless, and even when they aren't they are
// usually a mistake.
package assign
