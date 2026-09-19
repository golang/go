// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specexpr

import (
	"cmp"
	"fmt"
	"strings"
)

// Num represents a number in the solver. This is abstracted because we work
// with both concrete numbers and *symbolic widths* and need to be able to mix
// them. A Num is also an [Expr].
type Num interface {
	Expr

	// ValidWidth returns true if this is a valid width: either a fixed width
	// (128, 256, 512) or the scalable width VW.
	ValidWidth() bool

	Mul(x Num) (Num, error)
	Div(x Num) (Num, error)
	Compare(o Num) (int, bool)

	String() string
}

// Int is an integer that satisfies [Num].
type Int int

func (w Int) ValidWidth() bool {
	switch w {
	case 128, 256, 512:
		return true
	}
	return false
}
func (w Int) Mul(x Num) (Num, error) {
	switch x := x.(type) {
	case Int:
		return w * x, nil
	case ScalableWidth:
		return mkWidth(int(w)*x.num, x.denom), nil
	}
	panic("unknown Num")
}
func (w Int) Div(x Num) (Num, error) {
	switch x := x.(type) {
	case Int:
		if x == 0 {
			return nil, fmt.Errorf("division by zero")
		}
		if w%x != 0 {
			return nil, fmt.Errorf("inexact division %d/%d", w, x)
		}
		return w / x, nil
	case ScalableWidth:
		// ScalableWidth is implicitly multiplied by VW. We have no way to
		// express the inverse of VW.
		return nil, fmt.Errorf("cannot divide Int by ScalableWidth")
	}
	panic("unknown Num")
}
func (w Int) Compare(o Num) (int, bool) {
	if o, ok := o.(Int); ok {
		return cmp.Compare(w, o), true
	}
	return 0, false
}
func (w Int) String() string {
	return fmt.Sprint(int(w))
}
func (w Int) eval(b *Bindings) (any, error) {
	return w, nil
}
func (w Int) preorder(yield func(Expr) bool) bool {
	return true
}

// An ScalableWidth represents a symbolic width relative to a fixed but unknown
// scalable vector width VW. This is represented as a rational factor
// VW*num/denom
type ScalableWidth struct {
	num, denom int
}

// VW returns the base ScalableWidth representing a full-width scalable vector.
func VW() ScalableWidth {
	return ScalableWidth{1, 1}
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	if a < 0 {
		return -a
	}
	return a
}

func mkWidth(num, denom int) ScalableWidth {
	if denom == 0 {
		panic("denominator cannot be zero")
	}
	g := gcd(num, denom)
	num /= g
	denom /= g
	if denom < 0 {
		num = -num
		denom = -denom
	}
	return ScalableWidth{num, denom}
}

func (w ScalableWidth) ValidWidth() bool {
	return w == ScalableWidth{1, 1}
}

func (w ScalableWidth) Mul(x Num) (Num, error) {
	switch x := x.(type) {
	case Int:
		return mkWidth(w.num*int(x), w.denom), nil
	case ScalableWidth:
		return nil, fmt.Errorf("cannot multiply two scalable widths")
	}
	panic("unknown Num")
}

func (w ScalableWidth) Div(x Num) (Num, error) {
	switch x := x.(type) {
	case Int:
		if x == 0 {
			return nil, fmt.Errorf("division by zero")
		}
		return mkWidth(w.num, w.denom*int(x)), nil
	case ScalableWidth:
		a, b := w.num*x.denom, w.denom*x.num
		if b == 0 {
			return nil, fmt.Errorf("division by zero")
		}
		if a%b != 0 {
			return nil, fmt.Errorf("inexact division %d/%d", w, x)
		}
		return Int(a / b), nil
	}
	panic("unknown Num")
}

func (w ScalableWidth) Compare(o Num) (int, bool) {
	if o, ok := o.(ScalableWidth); ok {
		return cmp.Compare(w.num*o.denom, o.num*w.denom), true
	}
	return 0, false
}

func (w ScalableWidth) String() string {
	var buf strings.Builder
	buf.WriteString("VW")
	if w.num > 1 {
		fmt.Fprintf(&buf, "*%d", w.num)
	}
	if w.denom > 1 {
		fmt.Fprintf(&buf, "/%d", w.denom)
	}
	return buf.String()
}

func (w ScalableWidth) eval(b *Bindings) (any, error) {
	return w, nil
}

func (w ScalableWidth) preorder(yield func(Expr) bool) bool {
	return true
}
