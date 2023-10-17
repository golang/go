// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Some indirect uses of types crashed gccgo, because it assumed that
// the size of the type was known before it had been computed.

package p

type S1 struct {
	p *[1]S3
	s [][1]S3
	m map[int][1]S3
	c chan [1]S3
	i interface { f([1]S3) [1]S3 }
	f func([1]S3) [1]S3
}

type S2 struct {
	p *struct { F S3 }
	s []struct { F S3 }
	m map[int]struct { F S3 }
	c chan struct { F S3 }
	i interface { f(struct { F S3 }) struct { F S3 } }
	f func(struct { F S3 } ) struct { F S3 }
}

type S3 struct {
	I int
}
