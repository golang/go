// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pkgbits implements low-level coding abstractions for Unified IR's
// (UIR) binary export data format.
//
// At a low-level, the exported objects of a package are encoded as a byte
// array. This array contains byte representations of primitive, potentially
// variable-length values, such as integers, booleans, strings, and constants.
//
// Additionally, the array may contain values which denote indices in the byte
// array itself. These are termed "relocations" and allow for references.
//
// The details of mapping high-level Go constructs to primitives are left to
// other packages.
package pkgbits
