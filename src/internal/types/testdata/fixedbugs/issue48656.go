// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[P *Q, Q any](P, Q) {
	_ = f[P]
}

func f2[P /* ERROR "instantiation cycle" */ *Q, Q any](P, Q) {
	_ = f2[*P]
}
