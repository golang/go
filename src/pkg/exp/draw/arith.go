// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

// A Point is an X, Y coordinate pair.
type Point struct {
	X, Y int;
}

// ZP is the zero Point.
var ZP Point

// A Rectangle contains the Points with Min.X <= X < Max.X, Min.Y <= Y < Max.Y.
type Rectangle struct {
	Min, Max Point;
}

// ZR is the zero Rectangle.
var ZR Rectangle

// Pt is shorthand for Point{X, Y}.
func Pt(X, Y int) Point	{ return Point{X, Y} }

// Rect is shorthand for Rectangle{Pt(x0, y0), Pt(x1, y1)}.
func Rect(x0, y0, x1, y1 int) Rectangle	{ return Rectangle{Point{x0, y0}, Point{x1, y1}} }

// Rpt is shorthand for Rectangle{min, max}.
func Rpt(min, max Point) Rectangle	{ return Rectangle{min, max} }

// Add returns the sum of p and q: Pt(p.X+q.X, p.Y+q.Y).
func (p Point) Add(q Point) Point	{ return Point{p.X + q.X, p.Y + q.Y} }

// Sub returns the difference of p and q: Pt(p.X-q.X, p.Y-q.Y).
func (p Point) Sub(q Point) Point	{ return Point{p.X - q.X, p.Y - q.Y} }

// Mul returns p scaled by k: Pt(p.X*k p.Y*k).
func (p Point) Mul(k int) Point	{ return Point{p.X * k, p.Y * k} }

// Div returns p divided by k: Pt(p.X/k, p.Y/k).
func (p Point) Div(k int) Point	{ return Point{p.X / k, p.Y / k} }

// Eq returns true if p and q are equal.
func (p Point) Eq(q Point) bool	{ return p.X == q.X && p.Y == q.Y }

// Inset returns the rectangle r inset by n: Rect(r.Min.X+n, r.Min.Y+n, r.Max.X-n, r.Max.Y-n).
func (r Rectangle) Inset(n int) Rectangle {
	return Rectangle{Point{r.Min.X + n, r.Min.Y + n}, Point{r.Max.X - n, r.Max.Y - n}}
}

// Add returns the rectangle r translated by p: Rpt(r.Min.Add(p), r.Max.Add(p)).
func (r Rectangle) Add(p Point) Rectangle	{ return Rectangle{r.Min.Add(p), r.Max.Add(p)} }

// Sub returns the rectangle r translated by -p: Rpt(r.Min.Sub(p), r.Max.Sub(p)).
func (r Rectangle) Sub(p Point) Rectangle	{ return Rectangle{r.Min.Sub(p), r.Max.Sub(p)} }

// Canon returns a canonical version of r: the returned rectangle
// has Min.X <= Max.X and Min.Y <= Max.Y.
func (r Rectangle) Canon() Rectangle {
	if r.Max.X < r.Min.X {
		r.Max.X = r.Min.X
	}
	if r.Max.Y < r.Min.Y {
		r.Max.Y = r.Min.Y
	}
	return r;
}

// Overlaps returns true if r and r1 cross; that is, it returns true if they share any point.
func (r Rectangle) Overlaps(r1 Rectangle) bool {
	return r.Min.X < r1.Max.X && r1.Min.X < r.Max.X &&
		r.Min.Y < r1.Max.Y && r1.Min.Y < r.Max.Y
}

// Empty retruns true if r contains no points.
func (r Rectangle) Empty() bool	{ return r.Max.X <= r.Min.X || r.Max.Y <= r.Min.Y }

// InRect returns true if all the points in r are also in r1.
func (r Rectangle) In(r1 Rectangle) bool {
	if r.Empty() {
		return true
	}
	if r1.Empty() {
		return false
	}
	return r1.Min.X <= r.Min.X && r.Max.X <= r1.Max.X &&
		r1.Min.Y <= r.Min.Y && r.Max.Y <= r1.Max.Y;
}

// Combine returns the smallest rectangle containing all points from r and from r1.
func (r Rectangle) Combine(r1 Rectangle) Rectangle {
	if r.Empty() {
		return r1
	}
	if r1.Empty() {
		return r
	}
	if r.Min.X > r1.Min.X {
		r.Min.X = r1.Min.X
	}
	if r.Min.Y > r1.Min.Y {
		r.Min.Y = r1.Min.Y
	}
	if r.Max.X < r1.Max.X {
		r.Max.X = r1.Max.X
	}
	if r.Max.Y < r1.Max.Y {
		r.Max.Y = r1.Max.Y
	}
	return r;
}

// Clip returns the largest rectangle containing only points shared by r and r1.
func (r Rectangle) Clip(r1 Rectangle) Rectangle {
	if r.Empty() {
		return r
	}
	if r1.Empty() {
		return r1
	}
	if r.Min.X < r1.Min.X {
		r.Min.X = r1.Min.X
	}
	if r.Min.Y < r1.Min.Y {
		r.Min.Y = r1.Min.Y
	}
	if r.Max.X > r1.Max.X {
		r.Max.X = r1.Max.X
	}
	if r.Max.Y > r1.Max.Y {
		r.Max.Y = r1.Max.Y
	}
	return r;
}

// Dx returns the width of the rectangle r: r.Max.X - r.Min.X.
func (r Rectangle) Dx() int	{ return r.Max.X - r.Min.X }

// Dy returns the width of the rectangle r: r.Max.Y - r.Min.Y.
func (r Rectangle) Dy() int	{ return r.Max.Y - r.Min.Y }
