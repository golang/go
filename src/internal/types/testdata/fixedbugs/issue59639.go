// -lang=go1.17

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[P /* ERROR "requires go1.18" */ interface{}](P) {}

var v func(int) = f /* ERROR "requires go1.18" */
