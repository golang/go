// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type I1 interface {
	int | any
}

type I2 interface {
	int | any
}
