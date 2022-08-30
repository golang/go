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
	{},     // want "redundant type from array, slice, or map composite literal"
	{1, 2}, // want "redundant type from array, slice, or map composite literal"
	{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = [...]T{
	{},     // want "redundant type from array, slice, or map composite literal"
	{1, 2}, // want "redundant type from array, slice, or map composite literal"
	{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []T{
	{},     // want "redundant type from array, slice, or map composite literal"
	{1, 2}, // want "redundant type from array, slice, or map composite literal"
	{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []T{
	{}, // want "redundant type from array, slice, or map composite literal"
	10: {1, 2}, // want "redundant type from array, slice, or map composite literal"
	20: {3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []struct {
	x, y int
}{
	{}, // want "redundant type from array, slice, or map composite literal"
	10: {1, 2}, // want "redundant type from array, slice, or map composite literal"
	20: {3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []interface{}{
	T{},
	10: T{1, 2},
	20: T{3, 4},
}

var _ = [][]int{
	{},     // want "redundant type from array, slice, or map composite literal"
	{1, 2}, // want "redundant type from array, slice, or map composite literal"
	{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = [][]int{
	([]int{}),
	([]int{1, 2}),
	{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = [][][]int{
	{}, // want "redundant type from array, slice, or map composite literal"
	{ // want "redundant type from array, slice, or map composite literal"
		{},           // want "redundant type from array, slice, or map composite literal"
		{0, 1, 2, 3}, // want "redundant type from array, slice, or map composite literal"
		{4, 5},       // want "redundant type from array, slice, or map composite literal"
	},
}

var _ = map[string]T{
	"foo": {},     // want "redundant type from array, slice, or map composite literal"
	"bar": {1, 2}, // want "redundant type from array, slice, or map composite literal"
	"bal": {3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = map[string]struct {
	x, y int
}{
	"foo": {},     // want "redundant type from array, slice, or map composite literal"
	"bar": {1, 2}, // want "redundant type from array, slice, or map composite literal"
	"bal": {3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = map[string]interface{}{
	"foo": T{},
	"bar": T{1, 2},
	"bal": T{3, 4},
}

var _ = map[string][]int{
	"foo": {},     // want "redundant type from array, slice, or map composite literal"
	"bar": {1, 2}, // want "redundant type from array, slice, or map composite literal"
	"bal": {3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = map[string][]int{
	"foo": ([]int{}),
	"bar": ([]int{1, 2}),
	"bal": {3, 4}, // want "redundant type from array, slice, or map composite literal"
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
	{0, 0, Point{4, 1}, []Point{{0, 0}, {1, 0}, {1, 0}, {1, 0}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	{1, 0, Point{1, 4}, []Point{{0, 0}, {0, 1}, {0, 1}, {0, 1}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	{2, 0, Point{4, 1}, []Point{{0, 0}, {1, 0}, {1, 0}, {1, 0}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	{3, 0, Point{1, 4}, []Point{{0, 0}, {0, 1}, {0, 1}, {0, 1}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
}

var _ = [42]*T{
	{},     // want "redundant type from array, slice, or map composite literal"
	{1, 2}, // want "redundant type from array, slice, or map composite literal"
	{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = [...]*T{
	{},     // want "redundant type from array, slice, or map composite literal"
	{1, 2}, // want "redundant type from array, slice, or map composite literal"
	{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []*T{
	{},     // want "redundant type from array, slice, or map composite literal"
	{1, 2}, // want "redundant type from array, slice, or map composite literal"
	{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []*T{
	{}, // want "redundant type from array, slice, or map composite literal"
	10: {1, 2}, // want "redundant type from array, slice, or map composite literal"
	20: {3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []*struct {
	x, y int
}{
	{}, // want "redundant type from array, slice, or map composite literal"
	10: {1, 2}, // want "redundant type from array, slice, or map composite literal"
	20: {3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []interface{}{
	&T{},
	10: &T{1, 2},
	20: &T{3, 4},
}

var _ = []*[]int{
	{},     // want "redundant type from array, slice, or map composite literal"
	{1, 2}, // want "redundant type from array, slice, or map composite literal"
	{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []*[]int{
	(&[]int{}),
	(&[]int{1, 2}),
	{3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = []*[]*[]int{
	{}, // want "redundant type from array, slice, or map composite literal"
	{ // want "redundant type from array, slice, or map composite literal"
		{},           // want "redundant type from array, slice, or map composite literal"
		{0, 1, 2, 3}, // want "redundant type from array, slice, or map composite literal"
		{4, 5},       // want "redundant type from array, slice, or map composite literal"
	},
}

var _ = map[string]*T{
	"foo": {},     // want "redundant type from array, slice, or map composite literal"
	"bar": {1, 2}, // want "redundant type from array, slice, or map composite literal"
	"bal": {3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = map[string]*struct {
	x, y int
}{
	"foo": {},     // want "redundant type from array, slice, or map composite literal"
	"bar": {1, 2}, // want "redundant type from array, slice, or map composite literal"
	"bal": {3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = map[string]interface{}{
	"foo": &T{},
	"bar": &T{1, 2},
	"bal": &T{3, 4},
}

var _ = map[string]*[]int{
	"foo": {},     // want "redundant type from array, slice, or map composite literal"
	"bar": {1, 2}, // want "redundant type from array, slice, or map composite literal"
	"bal": {3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var _ = map[string]*[]int{
	"foo": (&[]int{}),
	"bar": (&[]int{1, 2}),
	"bal": {3, 4}, // want "redundant type from array, slice, or map composite literal"
}

var pieces4 = []*Piece{
	{0, 0, Point{4, 1}, []Point{{0, 0}, {1, 0}, {1, 0}, {1, 0}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	{1, 0, Point{1, 4}, []Point{{0, 0}, {0, 1}, {0, 1}, {0, 1}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	{2, 0, Point{4, 1}, []Point{{0, 0}, {1, 0}, {1, 0}, {1, 0}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	{3, 0, Point{1, 4}, []Point{{0, 0}, {0, 1}, {0, 1}, {0, 1}}, nil, nil}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
}

var _ = map[T]T2{
	{1, 2}: {3, 4}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	{5, 6}: {7, 8}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
}

var _ = map[*T]*T2{
	{1, 2}: {3, 4}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
	{5, 6}: {7, 8}, // want "redundant type from array, slice, or map composite literal" "redundant type from array, slice, or map composite literal"
}
