// errorcheck

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const (
	c0   = 1 << 100
	c1   = c0 * c0
	c2   = c1 * c1
	c3   = c2 * c2 // ERROR "overflow"
	c4   = c3 * c3
	c5   = c4 * c4
	c6   = c5 * c5
	c7   = c6 * c6
	c8   = c7 * c7
	c9   = c8 * c8
	c10  = c9 * c9
	c11  = c10 * c10
	c12  = c11 * c11
	c13  = c12 * c12
	c14  = c13 * c13
	c15  = c14 * c14
	c16  = c15 * c15
	c17  = c16 * c16
	c18  = c17 * c17
	c19  = c18 * c18
	c20  = c19 * c19
	c21  = c20 * c20
	c22  = c21 * c21
	c23  = c22 * c22
	c24  = c23 * c23
	c25  = c24 * c24
	c26  = c25 * c25
	c27  = c26 * c26
	c28  = c27 * c27
	c29  = c28 * c28
	c30  = c29 * c29
	c31  = c30 * c30
	c32  = c31 * c31
	c33  = c32 * c32
	c34  = c33 * c33
	c35  = c34 * c34
	c36  = c35 * c35
	c37  = c36 * c36
	c38  = c37 * c37
	c39  = c38 * c38
	c40  = c39 * c39
	c41  = c40 * c40
	c42  = c41 * c41
	c43  = c42 * c42
	c44  = c43 * c43
	c45  = c44 * c44
	c46  = c45 * c45
	c47  = c46 * c46
	c48  = c47 * c47
	c49  = c48 * c48
	c50  = c49 * c49
	c51  = c50 * c50
	c52  = c51 * c51
	c53  = c52 * c52
	c54  = c53 * c53
	c55  = c54 * c54
	c56  = c55 * c55
	c57  = c56 * c56
	c58  = c57 * c57
	c59  = c58 * c58
	c60  = c59 * c59
	c61  = c60 * c60
	c62  = c61 * c61
	c63  = c62 * c62
	c64  = c63 * c63
	c65  = c64 * c64
	c66  = c65 * c65
	c67  = c66 * c66
	c68  = c67 * c67
	c69  = c68 * c68
	c70  = c69 * c69
	c71  = c70 * c70
	c72  = c71 * c71
	c73  = c72 * c72
	c74  = c73 * c73
	c75  = c74 * c74
	c76  = c75 * c75
	c77  = c76 * c76
	c78  = c77 * c77
	c79  = c78 * c78
	c80  = c79 * c79
	c81  = c80 * c80
	c82  = c81 * c81
	c83  = c82 * c82
	c84  = c83 * c83
	c85  = c84 * c84
	c86  = c85 * c85
	c87  = c86 * c86
	c88  = c87 * c87
	c89  = c88 * c88
	c90  = c89 * c89
	c91  = c90 * c90
	c92  = c91 * c91
	c93  = c92 * c92
	c94  = c93 * c93
	c95  = c94 * c94
	c96  = c95 * c95
	c97  = c96 * c96
	c98  = c97 * c97
	c99  = c98 * c98
	c100 = c99 * c99
)

func main() {
	println(c1 / c1)
	println(c2 / c2)
	println(c3 / c3)
	println(c4 / c4)
	println(c5 / c5)
	println(c6 / c6)
	println(c7 / c7)
	println(c8 / c8)
	println(c9 / c9)
	println(c10 / c10)
	println(c20 / c20)
	println(c30 / c30)
	println(c40 / c40)
	println(c50 / c50)
	println(c60 / c60)
	println(c70 / c70)
	println(c80 / c80)
	println(c90 / c90)
	println(c100 / c100)
}
