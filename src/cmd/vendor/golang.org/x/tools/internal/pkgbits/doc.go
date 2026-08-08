// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pkgbits implements low-level coding abstractions for
// Unified IR's export data format.
//
// At a low-level, a package is a collection of bitstream elements.
// Each element has a "kind" and a dense, non-negative index.
// Elements can be randomly accessed given their kind and index.
//
// Individual elements are sequences of variable-length values (e.g.,
// integers, booleans, strings, go/constant values, cross-references
// to other elements). Package pkgbits provides APIs for encoding and
// decoding these low-level values, but the details of mapping
// higher-level Go constructs into elements is left to higher-level
// abstractions.
//
// Elements may cross-reference each other with "relocations." For
// example, an element representing a pointer type has a relocation
// referring to the element type.
//
// Go constructs may be composed as a constellation of multiple
// elements. For example, a declared function may have one element to
// describe the object (e.g., its name, type, position), and a
// separate element to describe its function body. This allows readers
// some flexibility in efficiently seeking or re-reading data (e.g.,
// inlining requires re-reading the function body for each inlined
// call, without needing to re-read the object-level details).
//
// This is a copy of internal/pkgbits in the Go implementation.
package pkgbits
