// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[T interface{~map[string]int}](x T) {
	_ = x == nil
}

// simplified test case from issue

type PathParamsConstraint interface {
        ~map[string]string | ~[]struct{key, value string}
}

type PathParams[T PathParamsConstraint] struct {
	t T
}

func (pp *PathParams[T]) IsNil() bool {
	return pp.t == nil // this must succeed
}
