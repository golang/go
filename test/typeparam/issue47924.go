// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Cache[K any] struct{}

func (c Cache[K]) foo(x interface{}, f func(K) bool) {
	f(x.(K))
}

var _ Cache[int]
