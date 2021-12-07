// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"encoding/xml"
	"fmt"
	"io"
)

type T int

func (T) Scan(x fmt.ScanState, c byte) {} // want `should have signature Scan\(fmt\.ScanState, rune\) error`

func (T) Format(fmt.State, byte) {} // want `should have signature Format\(fmt.State, rune\)`

type U int

func (U) Format(byte) {} // no error: first parameter must be fmt.State to trigger check

func (U) GobDecode() {} // want `should have signature GobDecode\(\[\]byte\) error`

// Test rendering of type names such as xml.Encoder in diagnostic.
func (U) MarshalXML(*xml.Encoder) {} // want `method MarshalXML\(\*xml.Encoder\) should...`

func (U) UnmarshalXML(*xml.Decoder, xml.StartElement) error { // no error: signature matches xml.Unmarshaler
	return nil
}

func (U) WriteTo(w io.Writer) {} // want `method WriteTo\(w io.Writer\) should have signature WriteTo\(io.Writer\) \(int64, error\)`

func (T) WriteTo(w io.Writer, more, args int) {} // ok - clearly not io.WriterTo

type I interface {
	ReadByte() byte // want `should have signature ReadByte\(\) \(byte, error\)`
}

type V int // V does not implement error.

func (V) As() T       { return 0 }     // ok - V is not an error
func (V) Is() bool    { return false } // ok - V is not an error
func (V) Unwrap() int { return 0 }     // ok - V is not an error

type E int

func (E) Error() string { return "" } // E implements error.

func (E) As()     {} // want `method As\(\) should have signature As\((any|interface\{\})\) bool`
func (E) Is()     {} // want `method Is\(\) should have signature Is\(error\) bool`
func (E) Unwrap() {} // want `method Unwrap\(\) should have signature Unwrap\(\) error`

type F int

func (F) Error() string { return "" } // Both F and *F implement error.

func (*F) As()     {} // want `method As\(\) should have signature As\((any|interface\{\})\) bool`
func (*F) Is()     {} // want `method Is\(\) should have signature Is\(error\) bool`
func (*F) Unwrap() {} // want `method Unwrap\(\) should have signature Unwrap\(\) error`

type G int

func (G) As(interface{}) bool // ok
