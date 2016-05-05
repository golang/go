// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	ok := true
	for _, tt := range tests {
		func() {
			defer func() {
				if err := recover(); err == nil {
					println(tt.name, "did not panic")
					ok = false
				}
			}()
			tt.fn()
		}()
	}
	if !ok {
		println("BUG")
	}
}

var intp *int
var slicep *[]byte
var a10p *[10]int
var a10Mp *[1<<20]int
var structp *Struct
var bigstructp *BigStruct
var i int
var m *M
var m1 *M1
var m2 *M2

func use(interface{}) {
}

var tests = []struct{
	name string
	fn func()
}{
	// Edit .+1,/^}/s/^[^	].+/	{"&", func() { println(&) }},\n	{"\&&", func() { println(\&&) }},/g
	{"*intp", func() { println(*intp) }},
	{"&*intp", func() { println(&*intp) }},
	{"*slicep", func() { println(*slicep) }},
	{"&*slicep", func() { println(&*slicep) }},
	{"(*slicep)[0]", func() { println((*slicep)[0]) }},
	{"&(*slicep)[0]", func() { println(&(*slicep)[0]) }},
	{"(*slicep)[i]", func() { println((*slicep)[i]) }},
	{"&(*slicep)[i]", func() { println(&(*slicep)[i]) }},
	{"*a10p", func() { use(*a10p) }},
	{"&*a10p", func() { println(&*a10p) }},
	{"a10p[0]", func() { println(a10p[0]) }},
	{"&a10p[0]", func() { println(&a10p[0]) }},
	{"a10p[i]", func() { println(a10p[i]) }},
	{"&a10p[i]", func() { println(&a10p[i]) }},
	{"*structp", func() { use(*structp) }},
	{"&*structp", func() { println(&*structp) }},
	{"structp.i", func() { println(structp.i) }},
	{"&structp.i", func() { println(&structp.i) }},
	{"structp.j", func() { println(structp.j) }},
	{"&structp.j", func() { println(&structp.j) }},
	{"structp.k", func() { println(structp.k) }},
	{"&structp.k", func() { println(&structp.k) }},
	{"structp.x[0]", func() { println(structp.x[0]) }},
	{"&structp.x[0]", func() { println(&structp.x[0]) }},
	{"structp.x[i]", func() { println(structp.x[i]) }},
	{"&structp.x[i]", func() { println(&structp.x[i]) }},
	{"structp.x[9]", func() { println(structp.x[9]) }},
	{"&structp.x[9]", func() { println(&structp.x[9]) }},
	{"structp.l", func() { println(structp.l) }},
	{"&structp.l", func() { println(&structp.l) }},
	{"*bigstructp", func() { use(*bigstructp) }},
	{"&*bigstructp", func() { println(&*bigstructp) }},
	{"bigstructp.i", func() { println(bigstructp.i) }},
	{"&bigstructp.i", func() { println(&bigstructp.i) }},
	{"bigstructp.j", func() { println(bigstructp.j) }},
	{"&bigstructp.j", func() { println(&bigstructp.j) }},
	{"bigstructp.k", func() { println(bigstructp.k) }},
	{"&bigstructp.k", func() { println(&bigstructp.k) }},
	{"bigstructp.x[0]", func() { println(bigstructp.x[0]) }},
	{"&bigstructp.x[0]", func() { println(&bigstructp.x[0]) }},
	{"bigstructp.x[i]", func() { println(bigstructp.x[i]) }},
	{"&bigstructp.x[i]", func() { println(&bigstructp.x[i]) }},
	{"bigstructp.x[9]", func() { println(bigstructp.x[9]) }},
	{"&bigstructp.x[9]", func() { println(&bigstructp.x[9]) }},
	{"bigstructp.x[100<<20]", func() { println(bigstructp.x[100<<20]) }},
	{"&bigstructp.x[100<<20]", func() { println(&bigstructp.x[100<<20]) }},
	{"bigstructp.l", func() { println(bigstructp.l) }},
	{"&bigstructp.l", func() { println(&bigstructp.l) }},
	{"m1.F()", func() { println(m1.F()) }},
	{"m1.M.F()", func() { println(m1.M.F()) }},
	{"m2.F()", func() { println(m2.F()) }},
	{"m2.M.F()", func() { println(m2.M.F()) }},
}

type Struct struct {
	i int
	j float64
	k string
	x [10]int
	l []byte
}

type BigStruct struct {
	i int
	j float64
	k string
	x [128<<20]byte
	l []byte
}

type M struct {
}

func (m *M) F() int {return 0}

type M1 struct {
	M
}

type M2 struct {
	x int
	M
}
