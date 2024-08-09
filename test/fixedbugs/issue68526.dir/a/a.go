// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.aliastypeparams

package a

// TODO(#68778): enable once type parameterized aliases are allowed in exportdata.
// type A[T any] = struct{ F T }

type B = struct{ F int }

func F() B {
	type a[T any] = struct{ F T }
	return a[int]{}
}
