// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ifaceassert defines an Analyzer that flags
// impossible interface-interface type assertions.
//
// # Analyzer ifaceassert
//
// ifaceassert: detect impossible interface-to-interface type assertions
//
// This checker flags type assertions v.(T) and corresponding type-switch cases
// in which the static type V of v is an interface that cannot possibly implement
// the target interface T. This occurs when V and T contain methods with the same
// name but different signatures. Example:
//
//	var v interface {
//		Read()
//	}
//	_ = v.(io.Reader)
//
// The Read method in v has a different signature than the Read method in
// io.Reader, so this assertion cannot succeed.
package ifaceassert
