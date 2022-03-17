// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Builder /* ERROR illegal cycle */ [T interface{ struct{ Builder[T] } }] struct{}
type myBuilder struct {
	Builder[myBuilder]
}
