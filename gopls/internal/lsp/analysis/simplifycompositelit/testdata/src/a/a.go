// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testdata

type T struct {
	x, y int
}

type T2 struct {
	w, z int
}

var _ = [42]T{
	T{},     // want "redundant type from array, slice, or map composite literal"
	T{1, 2}, // want "redundant type from array, slice, or map composite literal"
	T{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = [...]T{
	T{},     // want "redundant type from array, slice, or map composite literal"
	T{1, 2}, // want "redundant type from array, slice, or map composite literal"
	T{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []T{
	T{},     // want "redundant type from array, slice, or map composite literal"
	T{1, 2}, // want "redundant type from array, slice, or map composite literal"
	T{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []T{
	T{}, // want "redundant type from array, slice, or map composite literal"
	10:  T{1, 2}, // want "redundant type from array, slice, or map composite literal"
	20:  T{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []struct {
	x, y int
}{
	struct{ x, y int }{}, // want "redundant type from array, slice, or map composite literal"
	10:                   struct{ x, y int }{1, 2}, // want "redundant type from array, slice, or map composite literal"
	20:                   struct{ x, y int }{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []interface{}{
	T{},
	10: T{1, 2},
	20: T{3, 4},
}

var _ = [][]int{
	[]int{},     // want "redundant type from array, slice, or map composite literal"
	[]int{1, 2}, // want "redundant type from array, slice, or map composite literal"
	[]int{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = [][]int{
	([]int{}),
	([]int{1, 2}),
	[]int{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = [][][]int{
	[][]int{}, // want "redundant type from array, slice, or map composite literal"
	[][]int{ // want "redundant type from array, slice, or map composite literal"
		[]int{},           // want "redundant type from array, slice, or map composite literal"
		[]int{0, 1, 2, 3}, // want "redundant type from array, slice, or map composite literal"
		[]int{4, 5},       // want "redundant type from array, slice, or map composite literal"
	},
}

var _ = map[string]T{
	"foo": T{},     // want "redundant type from array, slice, or map composite literal"
	"bar": T{1, 2}, // want "redundant type from array, slice, or map composite literal"
	"bal": T{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = map[string]struct {
	x, y int
}{
	"foo": struct{ x, y int }{},     // want "redundant type from array, slice, or map composite literal"
	"bar": struct{ x, y int }{1, 2}, // want "redundant type from array, slice, or map composite literal"
	"bal": struct{ x, y int }{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = map[string]interface{}{
	"foo": T{},
	"bar": T{1, 2},
	"bal": T{3, 4},
}

var _ = map[string][]int{
	"foo": []int{},     // want "redundant type from array, slice, or map composite literal"
	"bar": []int{1, 2}, // want "redundant type from array, slice, or map composite literal"
	"bal": []int{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = map[string][]int{
	"foo": ([]int{}),
	"bar": ([]int{1, 2}),
	"bal": []int{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

type Point struct {
	a int
	b int
}

type Piece struct {
	a int
	b int
	c Point
	d []Point
	e *Point
	f *Point
}

// from exp/4s/data.go
var pieces3 = []Piece{
	Piece{0, 0, Point{4, 1}, []Point{Point{0, 0}, Point{1, 0}, Point{1, 0}, Point{1, 0}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	Piece{1, 0, Point{1, 4}, []Point{Point{0, 0}, Point{0, 1}, Point{0, 1}, Point{0, 1}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	Piece{2, 0, Point{4, 1}, []Point{Point{0, 0}, Point{1, 0}, Point{1, 0}, Point{1, 0}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	Piece{3, 0, Point{1, 4}, []Point{Point{0, 0}, Point{0, 1}, Point{0, 1}, Point{0, 1}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
}

var _ = [42]*T{
	&T{},     // want "redundant type from array, slice, or map composite literal"
	&T{1, 2}, // want "redundant type from array, slice, or map composite literal"
	&T{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = [...]*T{
	&T{},     // want "redundant type from array, slice, or map composite literal"
	&T{1, 2}, // want "redundant type from array, slice, or map composite literal"
	&T{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []*T{
	&T{},     // want "redundant type from array, slice, or map composite literal"
	&T{1, 2}, // want "redundant type from array, slice, or map composite literal"
	&T{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []*T{
	&T{}, // want "redundant type from array, slice, or map composite literal"
	10:   &T{1, 2}, // want "redundant type from array, slice, or map composite literal"
	20:   &T{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []*struct {
	x, y int
}{
	&struct{ x, y int }{}, // want "redundant type from array, slice, or map composite literal"
	10:                    &struct{ x, y int }{1, 2}, // want "redundant type from array, slice, or map composite literal"
	20:                    &struct{ x, y int }{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []interface{}{
	&T{},
	10: &T{1, 2},
	20: &T{3, 4},
}

var _ = []*[]int{
	&[]int{},     // want "redundant type from array, slice, or map composite literal"
	&[]int{1, 2}, // want "redundant type from array, slice, or map composite literal"
	&[]int{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []*[]int{
	(&[]int{}),
	(&[]int{1, 2}),
	&[]int{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []*[]*[]int{
	&[]*[]int{}, // want "redundant type from array, slice, or map composite literal"
	&[]*[]int{ // want "redundant type from array, slice, or map composite literal"
		&[]int{},           // want "redundant type from array, slice, or map composite literal"
		&[]int{0, 1, 2, 3}, // want "redundant type from array, slice, or map composite literal"
		&[]int{4, 5},       // want "redundant type from array, slice, or map composite literal"
	},
}

var _ = map[string]*T{
	"foo": &T{},     // want "redundant type from array, slice, or map composite literal"
	"bar": &T{1, 2}, // want "redundant type from array, slice, or map composite literal"
	"bal": &T{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = map[string]*struct {
	x, y int
}{
	"foo": &struct{ x, y int }{},     // want "redundant type from array, slice, or map composite literal"
	"bar": &struct{ x, y int }{1, 2}, // want "redundant type from array, slice, or map composite literal"
	"bal": &struct{ x, y int }{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = map[string]interface{}{
	"foo": &T{},
	"bar": &T{1, 2},
	"bal": &T{3, 4},
}

var _ = map[string]*[]int{
	"foo": &[]int{},     // want "redundant type from array, slice, or map composite literal"
	"bar": &[]int{1, 2}, // want "redundant type from array, slice, or map composite literal"
	"bal": &[]int{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = map[string]*[]int{
	"foo": (&[]int{}),
	"bar": (&[]int{1, 2}),
	"bal": &[]int{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var pieces4 = []*Piece{
	&Piece{0, 0, Point{4, 1}, []Point{Point{0, 0}, Point{1, 0}, Point{1, 0}, Point{1, 0}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	&Piece{1, 0, Point{1, 4}, []Point{Point{0, 0}, Point{0, 1}, Point{0, 1}, Point{0, 1}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	&Piece{2, 0, Point{4, 1}, []Point{Point{0, 0}, Point{1, 0}, Point{1, 0}, Point{1, 0}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	&Piece{3, 0, Point{1, 4}, []Point{Point{0, 0}, Point{0, 1}, Point{0, 1}, Point{0, 1}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
}

var _ = map[T]T2{
	T{1, 2}: T2{3, 4}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	T{5, 6}: T2{7, 8}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
}

var _ = map[*T]*T2{
	&T{1, 2}: &T2{3, 4}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	&T{5, 6}: &T2{7, 8}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
}
