// Copyright (c) 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64 || purego

package field

func (v *Element) carryPropagate() *Element {
	return v.carryPropagateGeneric()
}
