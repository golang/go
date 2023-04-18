// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type M interface {
	m()
}

type C interface {
	comparable
}

type _ interface {
	int | M          // ERROR "cannot use p.M in union (p.M contains methods)"
	int | comparable // ERROR "cannot use comparable in union"
	int | C          // ERROR "cannot use p.C in union (p.C embeds comparable)"
}
