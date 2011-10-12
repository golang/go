// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Semi-exhaustive test for append()

package main

import (
	"fmt"
	"reflect"
)


func verify(name string, result, expected interface{}) {
	if !reflect.DeepEqual(result, expected) {
		panic(name)
	}
}


func main() {
	for _, t := range tests {
		verify(t.name, t.result, t.expected)
	}
	verifyStruct()
	verifyInterface()
}


var tests = []struct {
	name             string
	result, expected interface{}
}{
	{"bool a", append([]bool{}), []bool{}},
	{"bool b", append([]bool{}, true), []bool{true}},
	{"bool c", append([]bool{}, true, false, true, true), []bool{true, false, true, true}},

	{"bool d", append([]bool{true, false, true}), []bool{true, false, true}},
	{"bool e", append([]bool{true, false, true}, false), []bool{true, false, true, false}},
	{"bool f", append([]bool{true, false, true}, false, false, false), []bool{true, false, true, false, false, false}},

	{"bool g", append([]bool{}, []bool{true}...), []bool{true}},
	{"bool h", append([]bool{}, []bool{true, false, true, false}...), []bool{true, false, true, false}},

	{"bool i", append([]bool{true, false, true}, []bool{true}...), []bool{true, false, true, true}},
	{"bool j", append([]bool{true, false, true}, []bool{true, true, true}...), []bool{true, false, true, true, true, true}},


	{"byte a", append([]byte{}), []byte{}},
	{"byte b", append([]byte{}, 0), []byte{0}},
	{"byte c", append([]byte{}, 0, 1, 2, 3), []byte{0, 1, 2, 3}},

	{"byte d", append([]byte{0, 1, 2}), []byte{0, 1, 2}},
	{"byte e", append([]byte{0, 1, 2}, 3), []byte{0, 1, 2, 3}},
	{"byte f", append([]byte{0, 1, 2}, 3, 4, 5), []byte{0, 1, 2, 3, 4, 5}},

	{"byte g", append([]byte{}, []byte{0}...), []byte{0}},
	{"byte h", append([]byte{}, []byte{0, 1, 2, 3}...), []byte{0, 1, 2, 3}},

	{"byte i", append([]byte{0, 1, 2}, []byte{3}...), []byte{0, 1, 2, 3}},
	{"byte j", append([]byte{0, 1, 2}, []byte{3, 4, 5}...), []byte{0, 1, 2, 3, 4, 5}},

	{"bytestr a", append([]byte{}, "0"...), []byte("0")},
	{"bytestr b", append([]byte{}, "0123"...), []byte("0123")},

	{"bytestr c", append([]byte("012"), "3"...), []byte("0123")},
	{"bytestr d", append([]byte("012"), "345"...), []byte("012345")},

	{"int16 a", append([]int16{}), []int16{}},
	{"int16 b", append([]int16{}, 0), []int16{0}},
	{"int16 c", append([]int16{}, 0, 1, 2, 3), []int16{0, 1, 2, 3}},

	{"int16 d", append([]int16{0, 1, 2}), []int16{0, 1, 2}},
	{"int16 e", append([]int16{0, 1, 2}, 3), []int16{0, 1, 2, 3}},
	{"int16 f", append([]int16{0, 1, 2}, 3, 4, 5), []int16{0, 1, 2, 3, 4, 5}},

	{"int16 g", append([]int16{}, []int16{0}...), []int16{0}},
	{"int16 h", append([]int16{}, []int16{0, 1, 2, 3}...), []int16{0, 1, 2, 3}},

	{"int16 i", append([]int16{0, 1, 2}, []int16{3}...), []int16{0, 1, 2, 3}},
	{"int16 j", append([]int16{0, 1, 2}, []int16{3, 4, 5}...), []int16{0, 1, 2, 3, 4, 5}},


	{"uint32 a", append([]uint32{}), []uint32{}},
	{"uint32 b", append([]uint32{}, 0), []uint32{0}},
	{"uint32 c", append([]uint32{}, 0, 1, 2, 3), []uint32{0, 1, 2, 3}},

	{"uint32 d", append([]uint32{0, 1, 2}), []uint32{0, 1, 2}},
	{"uint32 e", append([]uint32{0, 1, 2}, 3), []uint32{0, 1, 2, 3}},
	{"uint32 f", append([]uint32{0, 1, 2}, 3, 4, 5), []uint32{0, 1, 2, 3, 4, 5}},

	{"uint32 g", append([]uint32{}, []uint32{0}...), []uint32{0}},
	{"uint32 h", append([]uint32{}, []uint32{0, 1, 2, 3}...), []uint32{0, 1, 2, 3}},

	{"uint32 i", append([]uint32{0, 1, 2}, []uint32{3}...), []uint32{0, 1, 2, 3}},
	{"uint32 j", append([]uint32{0, 1, 2}, []uint32{3, 4, 5}...), []uint32{0, 1, 2, 3, 4, 5}},


	{"float64 a", append([]float64{}), []float64{}},
	{"float64 b", append([]float64{}, 0), []float64{0}},
	{"float64 c", append([]float64{}, 0, 1, 2, 3), []float64{0, 1, 2, 3}},

	{"float64 d", append([]float64{0, 1, 2}), []float64{0, 1, 2}},
	{"float64 e", append([]float64{0, 1, 2}, 3), []float64{0, 1, 2, 3}},
	{"float64 f", append([]float64{0, 1, 2}, 3, 4, 5), []float64{0, 1, 2, 3, 4, 5}},

	{"float64 g", append([]float64{}, []float64{0}...), []float64{0}},
	{"float64 h", append([]float64{}, []float64{0, 1, 2, 3}...), []float64{0, 1, 2, 3}},

	{"float64 i", append([]float64{0, 1, 2}, []float64{3}...), []float64{0, 1, 2, 3}},
	{"float64 j", append([]float64{0, 1, 2}, []float64{3, 4, 5}...), []float64{0, 1, 2, 3, 4, 5}},


	{"complex128 a", append([]complex128{}), []complex128{}},
	{"complex128 b", append([]complex128{}, 0), []complex128{0}},
	{"complex128 c", append([]complex128{}, 0, 1, 2, 3), []complex128{0, 1, 2, 3}},

	{"complex128 d", append([]complex128{0, 1, 2}), []complex128{0, 1, 2}},
	{"complex128 e", append([]complex128{0, 1, 2}, 3), []complex128{0, 1, 2, 3}},
	{"complex128 f", append([]complex128{0, 1, 2}, 3, 4, 5), []complex128{0, 1, 2, 3, 4, 5}},

	{"complex128 g", append([]complex128{}, []complex128{0}...), []complex128{0}},
	{"complex128 h", append([]complex128{}, []complex128{0, 1, 2, 3}...), []complex128{0, 1, 2, 3}},

	{"complex128 i", append([]complex128{0, 1, 2}, []complex128{3}...), []complex128{0, 1, 2, 3}},
	{"complex128 j", append([]complex128{0, 1, 2}, []complex128{3, 4, 5}...), []complex128{0, 1, 2, 3, 4, 5}},


	{"string a", append([]string{}), []string{}},
	{"string b", append([]string{}, "0"), []string{"0"}},
	{"string c", append([]string{}, "0", "1", "2", "3"), []string{"0", "1", "2", "3"}},

	{"string d", append([]string{"0", "1", "2"}), []string{"0", "1", "2"}},
	{"string e", append([]string{"0", "1", "2"}, "3"), []string{"0", "1", "2", "3"}},
	{"string f", append([]string{"0", "1", "2"}, "3", "4", "5"), []string{"0", "1", "2", "3", "4", "5"}},

	{"string g", append([]string{}, []string{"0"}...), []string{"0"}},
	{"string h", append([]string{}, []string{"0", "1", "2", "3"}...), []string{"0", "1", "2", "3"}},

	{"string i", append([]string{"0", "1", "2"}, []string{"3"}...), []string{"0", "1", "2", "3"}},
	{"string j", append([]string{"0", "1", "2"}, []string{"3", "4", "5"}...), []string{"0", "1", "2", "3", "4", "5"}},
}


func verifyStruct() {
	type T struct {
		a, b, c string
	}
	type S []T
	e := make(S, 100)
	for i := range e {
		e[i] = T{"foo", fmt.Sprintf("%d", i), "bar"}
	}

	verify("struct a", append(S{}), S{})
	verify("struct b", append(S{}, e[0]), e[0:1])
	verify("struct c", append(S{}, e[0], e[1], e[2]), e[0:3])

	verify("struct d", append(e[0:1]), e[0:1])
	verify("struct e", append(e[0:1], e[1]), e[0:2])
	verify("struct f", append(e[0:1], e[1], e[2], e[3]), e[0:4])

	verify("struct g", append(e[0:3]), e[0:3])
	verify("struct h", append(e[0:3], e[3]), e[0:4])
	verify("struct i", append(e[0:3], e[3], e[4], e[5], e[6]), e[0:7])

	for i := range e {
		verify("struct j", append(S{}, e[0:i]...), e[0:i])
		input := make(S, i)
		copy(input, e[0:i])
		verify("struct k", append(input, e[i:]...), e)
		verify("struct k - input modified", input, e[0:i])
	}

	s := make(S, 10, 20)
	r := make(S, len(s)+len(e))
	for i, x := range e {
		r[len(s)+i] = x
	}
	verify("struct l", append(s), s)
	verify("struct m", append(s, e...), r)
}


func verifyInterface() {
	type T interface{}
	type S []T
	e := make(S, 100)
	for i := range e {
		switch i % 4 {
		case 0:
			e[i] = i
		case 1:
			e[i] = "foo"
		case 2:
			e[i] = fmt.Sprintf("%d", i)
		case 3:
			e[i] = float64(i)
		}
	}

	verify("interface a", append(S{}), S{})
	verify("interface b", append(S{}, e[0]), e[0:1])
	verify("interface c", append(S{}, e[0], e[1], e[2]), e[0:3])

	verify("interface d", append(e[0:1]), e[0:1])
	verify("interface e", append(e[0:1], e[1]), e[0:2])
	verify("interface f", append(e[0:1], e[1], e[2], e[3]), e[0:4])

	verify("interface g", append(e[0:3]), e[0:3])
	verify("interface h", append(e[0:3], e[3]), e[0:4])
	verify("interface i", append(e[0:3], e[3], e[4], e[5], e[6]), e[0:7])

	for i := range e {
		verify("interface j", append(S{}, e[0:i]...), e[0:i])
		input := make(S, i)
		copy(input, e[0:i])
		verify("interface k", append(input, e[i:]...), e)
		verify("interface k - input modified", input, e[0:i])
	}

	s := make(S, 10, 20)
	r := make(S, len(s)+len(e))
	for i, x := range e {
		r[len(s)+i] = x
	}
	verify("interface l", append(s), s)
	verify("interface m", append(s, e...), r)
}
