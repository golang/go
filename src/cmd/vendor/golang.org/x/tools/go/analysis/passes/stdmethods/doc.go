// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package stdmethods defines an Analyzer that checks for misspellings
// in the signatures of methods similar to well-known interfaces.
//
// # Analyzer stdmethods
//
// stdmethods: check signature of methods of well-known interfaces
//
// Sometimes a type may be intended to satisfy an interface but may fail to
// do so because of a mistake in its method signature.
// For example, the result of this WriteTo method should be (int64, error),
// not error, to satisfy io.WriterTo:
//
//	type myWriterTo struct{...}
//	func (myWriterTo) WriteTo(w io.Writer) error { ... }
//
// This check ensures that each method whose name matches one of several
// well-known interface methods from the standard library has the correct
// signature for that interface.
//
// Checked method names include:
//
//	Format GobEncode GobDecode MarshalJSON MarshalXML
//	Peek ReadByte ReadFrom ReadRune Scan Seek
//	UnmarshalJSON UnreadByte UnreadRune WriteByte
//	WriteTo
package stdmethods
