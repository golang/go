// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

type Service uint64

var q *Service
var r *Service

type f struct{}

var fk f

func No(s a.A, qq uint8) *Service {
	defer func() { q, r = r, q }()
	return q
}

func Yes(s a.A, p *uint64) a.A {
	return a.V(s, fk, p)
}
