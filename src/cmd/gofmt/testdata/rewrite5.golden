//gofmt -r=x+x->2*x

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Rewriting of expressions containing nodes with associated comments to
// expressions without those nodes must also eliminate the associated
// comments.

package p

func f(x int) int {
	_ = 2 * x // this comment remains in the rewrite
	_ = 2 * x
	return 2 * x
}
