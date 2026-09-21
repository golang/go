// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scan

// FilterNilAVX512 is the simd version of FilterNil,
// it is implemented in assembly.
func FilterNilAVX512(bufp *uintptr, n int32) int32
