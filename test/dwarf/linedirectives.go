// run

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//line foo/bar.y:4
package main
//line foo/bar.y:60
func main() { 
//line foo/bar.y:297
	f, l := 0, 0
//line yacctab:1
	f, l = 1, 1
//line yaccpar:1
	f, l = 2, 1
//line foo/bar.y:82
	f, l = 3, 82
//line foo/bar.y:90
	f, l = 3, 90
//line foo/bar.y:92
	f, l = 3, 92
//line foo/bar.y:100
	f, l = 3, 100
//line foo/bar.y:104
	l = 104
//line foo/bar.y:112
	l = 112
//line foo/bar.y:117
	l = 117
//line foo/bar.y:121
	l = 121
//line foo/bar.y:125
	l = 125
//line foo/bar.y:133
	l = 133
//line foo/bar.y:146
	l = 146
//line foo/bar.y:148
//line foo/bar.y:153
//line foo/bar.y:155
	l = 155
//line foo/bar.y:160

//line foo/bar.y:164
//line foo/bar.y:173

//line foo/bar.y:178
//line foo/bar.y:180
//line foo/bar.y:185
//line foo/bar.y:195
//line foo/bar.y:197
//line foo/bar.y:202
//line foo/bar.y:204
//line foo/bar.y:208
//line foo/bar.y:211
//line foo/bar.y:213
//line foo/bar.y:215
//line foo/bar.y:217
//line foo/bar.y:221
//line foo/bar.y:229
//line foo/bar.y:236
//line foo/bar.y:238
//line foo/bar.y:240
//line foo/bar.y:244
//line foo/bar.y:249
//line foo/bar.y:253
//line foo/bar.y:257
//line foo/bar.y:262
//line foo/bar.y:267
//line foo/bar.y:272
	if l == f {
//line foo/bar.y:277
	panic("aie!")
//line foo/bar.y:281
	}
//line foo/bar.y:285
	return
//line foo/bar.y:288
//line foo/bar.y:290
}
//line foo/bar.y:293
//line foo/bar.y:295
