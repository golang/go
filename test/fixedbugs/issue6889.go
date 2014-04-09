// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 6889: confusing error message: ovf in mpaddxx

package main

const (
	f1  = 1
	f2  = f1 * 2
	f3  = f2 * 3
	f4  = f3 * 4
	f5  = f4 * 5
	f6  = f5 * 6
	f7  = f6 * 7
	f8  = f7 * 8
	f9  = f8 * 9
	f10 = f9 * 10
	f11 = f10 * 11
	f12 = f11 * 12
	f13 = f12 * 13
	f14 = f13 * 14
	f15 = f14 * 15
	f16 = f15 * 16
	f17 = f16 * 17
	f18 = f17 * 18
	f19 = f18 * 19
	f20 = f19 * 20
	f21 = f20 * 21
	f22 = f21 * 22
	f23 = f22 * 23
	f24 = f23 * 24
	f25 = f24 * 25
	f26 = f25 * 26
	f27 = f26 * 27
	f28 = f27 * 28
	f29 = f28 * 29
	f30 = f29 * 30
	f31 = f30 * 31
	f32 = f31 * 32
	f33 = f32 * 33
	f34 = f33 * 34
	f35 = f34 * 35
	f36 = f35 * 36
	f37 = f36 * 37
	f38 = f37 * 38
	f39 = f38 * 39
	f40 = f39 * 40
	f41 = f40 * 41
	f42 = f41 * 42
	f43 = f42 * 43
	f44 = f43 * 44
	f45 = f44 * 45
	f46 = f45 * 46
	f47 = f46 * 47
	f48 = f47 * 48
	f49 = f48 * 49
	f50 = f49 * 50
	f51 = f50 * 51
	f52 = f51 * 52
	f53 = f52 * 53
	f54 = f53 * 54
	f55 = f54 * 55
	f56 = f55 * 56
	f57 = f56 * 57
	f58 = f57 * 58
	f59 = f58 * 59
	f60 = f59 * 60
	f61 = f60 * 61
	f62 = f61 * 62
	f63 = f62 * 63
	f64 = f63 * 64
	f65 = f64 * 65
	f66 = f65 * 66
	f67 = f66 * 67
	f68 = f67 * 68
	f69 = f68 * 69
	f70 = f69 * 70
	f71 = f70 * 71
	f72 = f71 * 72
	f73 = f72 * 73
	f74 = f73 * 74
	f75 = f74 * 75
	f76 = f75 * 76
	f77 = f76 * 77
	f78 = f77 * 78
	f79 = f78 * 79
	f80 = f79 * 80
	f81 = f80 * 81
	f82 = f81 * 82
	f83 = f82 * 83
	f84 = f83 * 84
	f85 = f84 * 85
	f86 = f85 * 86
	f87 = f86 * 87
	f88 = f87 * 88
	f89 = f88 * 89
	f90 = f89 * 90
	f91 = f90 * 91 // ERROR "overflow"
)
