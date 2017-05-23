package main

var x struct {
	a, b, c int64
	d       struct{ p, q, r int32 }
	e       [8]byte
	f       [4]struct{ p, q, r int32 }
}

var y = &x.b
var z = &x.d.q

var b [10]byte
var c = &b[5]

var w = &x.f[3].r
