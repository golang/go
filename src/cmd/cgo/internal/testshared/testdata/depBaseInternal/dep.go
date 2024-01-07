// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// depBaseInternal is only imported by depBase.

package depBaseInternal

var Initialized bool

func init() {
	Initialized = true
}
