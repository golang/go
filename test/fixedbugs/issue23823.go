// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// The compiler cannot handle this. Disabled for now.
// See issue #25838.
/*
type I1 = interface {
	I2
}

type I2 interface {
	I1
}
*/
