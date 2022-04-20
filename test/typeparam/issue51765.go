// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type empty[T any] struct{}

func (this *empty[T]) Next() (empty T, _ error) {
	return empty, nil
}

var _ = &empty[string]{}
