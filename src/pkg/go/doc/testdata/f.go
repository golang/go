// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The package f is a go/doc test for functions and factory methods.
package f

// ----------------------------------------------------------------------------
// Factory functions for non-exported types must not get lost.

type private struct{}

// Exported must always be visible. Was issue 2824.
func Exported() private {}
