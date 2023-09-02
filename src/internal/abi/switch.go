// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

type InterfaceSwitch struct {
	NCases int

	// Array of NCases elements.
	// Each case must be a non-empty interface type.
	Cases [1]*InterfaceType
}
