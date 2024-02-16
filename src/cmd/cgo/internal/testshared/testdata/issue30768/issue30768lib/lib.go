// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue30768lib

// S is a struct that requires a generated hash function.
type S struct {
	A string
	B int
}
