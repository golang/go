// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import "fmt"

type T[P any] int

func (T[_]) Scan(x fmt.ScanState, c byte) {} // want `should have signature Scan\(fmt\.ScanState, rune\) error`

func (T[_]) Format(fmt.State, byte) {} // want `should have signature Format\(fmt.State, rune\)`

type U[P any] int

func (U[_]) Format(byte) {} // no error: first parameter must be fmt.State to trigger check

func (U[P]) GobDecode(P) {} // want `should have signature GobDecode\(\[\]byte\) error`

type V[P any] int // V does not implement error.

func (V[_]) As() T[int]  { return 0 }     // ok - V is not an error
func (V[_]) Is() bool    { return false } // ok - V is not an error
func (V[_]) Unwrap() int { return 0 }     // ok - V is not an error

type E[P any] int

func (E[_]) Error() string { return "" } // E implements error.

func (E[P]) As()     {} // want `method As\(\) should have signature As\((any|interface\{\})\) bool`
func (E[_]) Is()     {} // want `method Is\(\) should have signature Is\(error\) bool`
func (E[_]) Unwrap() {} // want `method Unwrap\(\) should have signature Unwrap\(\) error`

type F[P any] int

func (F[_]) Error() string { return "" } // Both F and *F implement error.

func (*F[_]) As()     {} // want `method As\(\) should have signature As\((any|interface\{\})\) bool`
func (*F[_]) Is()     {} // want `method Is\(\) should have signature Is\(error\) bool`
func (*F[_]) Unwrap() {} // want `method Unwrap\(\) should have signature Unwrap\(\) error`
