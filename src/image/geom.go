// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package image

import (
	"image/color"
	"math/bits"
	"strconv"
)

// A Point is an X, Y coordinate pair. The axes increase right and down.
type Point struct {
	X, Y int
}

// String returns a string representation of p like "(3,4)".
func (p Point) String() string {
	return "(" + strconv.Itoa(p.X) + "," + strconv.Itoa(p.Y) + ")"
}

// Add returns the vector p+q.
func (p Point) Add(q Point) Point {
	return Point{p.X + q.X, p.Y + q.Y}
}

// Sub returns the vector p-q.
func (p Point) Sub(q Point) Point {
	return Point{p.X - q.X, p.Y - q.Y}
}

// Mul returns the vector p*k.
func (p Point) Mul(k int) Point {
	return Point{p.X * k, p.Y * k}
}

// Div returns the vector p/k.
func (p Point) Div(k int) Point {
	return Point{p.X / k, p.Y / k}
}

// In reports whether p is in r.
func (p Point) In(r Rectangle) bool {
	return r.Min.X <= p.X && p.X < r.Max.X &&
		r.Min.Y <= p.Y && p.Y < r.Max.Y
}

// Mod returns the point q in r such that p.X-q.X is a multiple of r's width
// and p.Y-q.Y is a multiple of r's height.
func (p Point) Mod(r Rectangle) Point {
	w, h := r.Dx(), r.Dy()
	p = p.Sub(r.Min)
	p.X = p.X % w
	if p.X < 0 {
		p.X += w
	}
	p.Y = p.Y % h
	if p.Y < 0 {
		p.Y += h
	}
	return p.Add(r.Min)
}

// Eq reports whether p and q are equal.
func (p Point) Eq(q Point) bool {
	return p == q
}

// ZP is the zero [Point].
//
// Deprecated: Use a literal [image.Point] instead.
var ZP Point

// Pt is shorthand for [Point]{X, Y}.
func Pt(X, Y int) Point {
	return Point{X, Y}
}

// A Rectangle contains the points with Min.X <= X < Max.X, Min.Y <= Y < Max.Y.
// It is well-formed if Min.X <= Max.X and likewise for Y. Points are always
// well-formed. A rectangle's methods always return well-formed outputs for
// well-formed inputs.
//
// A Rectangle is also an [Image] whose bounds are the rectangle itself. At
// returns color.Opaque for points in the rectangle and color.Transparent
// otherwise.
type Rectangle struct {
	Min, Max Point
}

// String returns a string representation of r like "(3,4)-(6,5)".
func (r Rectangle) String() string {
	return r.Min.String() + "-" + r.Max.String()
}

// Dx returns r's width.
func (r Rectangle) Dx() int {
	return r.Max.X - r.Min.X
}

// Dy returns r's height.
func (r Rectangle) Dy() int {
	return r.Max.Y - r.Min.Y
}

// Size returns r's width and height.
func (r Rectangle) Size() Point {
	return Point{
		r.Max.X - r.Min.X,
		r.Max.Y - r.Min.Y,
	}
}

// Add returns the rectangle r translated by p.
func (r Rectangle) Add(p Point) Rectangle {
	return Rectangle{
		Point{r.Min.X + p.X, r.Min.Y + p.Y},
		Point{r.Max.X + p.X, r.Max.Y + p.Y},
	}
}

// Sub returns the rectangle r translated by -p.
func (r Rectangle) Sub(p Point) Rectangle {
	return Rectangle{
		Point{r.Min.X - p.X, r.Min.Y - p.Y},
		Point{r.Max.X - p.X, r.Max.Y - p.Y},
	}
}

// Inset returns the rectangle r inset by n, which may be negative. If either
// of r's dimensions is less than 2*n then an empty rectangle near the center
// of r will be returned.
func (r Rectangle) Inset(n int) Rectangle {
	if r.Dx() < 2*n {
		r.Min.X = (r.Min.X + r.Max.X) / 2
		r.Max.X = r.Min.X
	} else {
		r.Min.X += n
		r.Max.X -= n
	}
	if r.Dy() < 2*n {
		r.Min.Y = (r.Min.Y + r.Max.Y) / 2
		r.Max.Y = r.Min.Y
	} else {
		r.Min.Y += n
		r.Max.Y -= n
	}
	return r
}

// Intersect returns the largest rectangle contained by both r and s. If the
// two rectangles do not overlap then the zero rectangle will be returned.
func (r Rectangle) Intersect(s Rectangle) Rectangle {
	if r.Min.X < s.Min.X {
		r.Min.X = s.Min.X
	}
	if r.Min.Y < s.Min.Y {
		r.Min.Y = s.Min.Y
	}
	if r.Max.X > s.Max.X {
		r.Max.X = s.Max.X
	}
	if r.Max.Y > s.Max.Y {
		r.Max.Y = s.Max.Y
	}
	// Letting r0 and s0 be the values of r and s at the time that the method
	// is called, this next line is equivalent to:
	//
	// if max(r0.Min.X, s0.Min.X) >= min(r0.Max.X, s0.Max.X) || likewiseForY { etc }
	if r.Empty() {
		return Rectangle{}
	}
	return r
}

// Union returns the smallest rectangle that contains both r and s.
func (r Rectangle) Union(s Rectangle) Rectangle {
	if r.Empty() {
		return s
	}
	if s.Empty() {
		return r
	}
	if r.Min.X > s.Min.X {
		r.Min.X = s.Min.X
	}
	if r.Min.Y > s.Min.Y {
		r.Min.Y = s.Min.Y
	}
	if r.Max.X < s.Max.X {
		r.Max.X = s.Max.X
	}
	if r.Max.Y < s.Max.Y {
		r.Max.Y = s.Max.Y
	}
	return r
}

// Empty reports whether the rectangle contains no points.
func (r Rectangle) Empty() bool {
	return r.Min.X >= r.Max.X || r.Min.Y >= r.Max.Y
}

// Eq reports whether r and s contain the same set of points. All empty
// rectangles are considered equal.
func (r Rectangle) Eq(s Rectangle) bool {
	return r == s || r.Empty() && s.Empty()
}

// Overlaps reports whether r and s have a non-empty intersection.
func (r Rectangle) Overlaps(s Rectangle) bool {
	return !r.Empty() && !s.Empty() &&
		r.Min.X < s.Max.X && s.Min.X < r.Max.X &&
		r.Min.Y < s.Max.Y && s.Min.Y < r.Max.Y
}

// In reports whether every point in r is in s.
func (r Rectangle) In(s Rectangle) bool {
	if r.Empty() {
		return true
	}
	// Note that r.Max is an exclusive bound for r, so that r.In(s)
	// does not require that r.Max.In(s).
	return s.Min.X <= r.Min.X && r.Max.X <= s.Max.X &&
		s.Min.Y <= r.Min.Y && r.Max.Y <= s.Max.Y
}

// Canon returns the canonical version of r. The returned rectangle has minimum
// and maximum coordinates swapped if necessary so that it is well-formed.
func (r Rectangle) Canon() Rectangle {
	if r.Max.X < r.Min.X {
		r.Min.X, r.Max.X = r.Max.X, r.Min.X
	}
	if r.Max.Y < r.Min.Y {
		r.Min.Y, r.Max.Y = r.Max.Y, r.Min.Y
	}
	return r
}

// At implements the [Image] interface.
func (r Rectangle) At(x, y int) color.Color {
	if (Point{x, y}).In(r) {
		return color.Opaque
	}
	return color.Transparent
}

// RGBA64At implements the [RGBA64Image] interface.
func (r Rectangle) RGBA64At(x, y int) color.RGBA64 {
	if (Point{x, y}).In(r) {
		return color.RGBA64{0xffff, 0xffff, 0xffff, 0xffff}
	}
	return color.RGBA64{}
}

// Bounds implements the [Image] interface.
func (r Rectangle) Bounds() Rectangle {
	return r
}

// ColorModel implements the [Image] interface.
func (r Rectangle) ColorModel() color.Model {
	return color.Alpha16Model
}

// ZR is the zero [Rectangle].
//
// Deprecated: Use a literal [image.Rectangle] instead.
var ZR Rectangle

// Rect is shorthand for [Rectangle]{Pt(x0, y0), [Pt](x1, y1)}. The returned
// rectangle has minimum and maximum coordinates swapped if necessary so that
// it is well-formed.
func Rect(x0, y0, x1, y1 int) Rectangle {
	if x0 > x1 {
		x0, x1 = x1, x0
	}
	if y0 > y1 {
		y0, y1 = y1, y0
	}
	return Rectangle{Point{x0, y0}, Point{x1, y1}}
}

// mul3NonNeg returns (x * y * z), unless at least one argument is negative or
// if the computation overflows the int type, in which case it returns -1.
func mul3NonNeg(x int, y int, z int) int {
	if (x < 0) || (y < 0) || (z < 0) {
		return -1
	}
	hi, lo := bits.Mul64(uint64(x), uint64(y))
	if hi != 0 {
		return -1
	}
	hi, lo = bits.Mul64(lo, uint64(z))
	if hi != 0 {
		return -1
	}
	a := int(lo)
	if (a < 0) || (uint64(a) != lo) {
		return -1
	}
	return a
}

// add2NonNeg returns (x + y), unless at least one argument is negative or if
// the computation overflows the int type, in which case it returns -1.
func add2NonNeg(x int, y int) int {
	if (x < 0) || (y < 0) {
		return -1
	}
	a := x + y
	if a < 0 {
		return -1
	}
	return a
}
