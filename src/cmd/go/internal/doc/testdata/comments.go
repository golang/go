// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkg

type InterfaceWithComments interface {
	// First group of methods.

	// First does something.
	First() // First line comment.

	// Hidden group of methods.

	hidden() // Hidden line comment.

	// Second group of methods.

	// Another heading.
	//ignore:directive

	Second()

	/* Embedded interfaces. */

	error

	// Trailing interface comment.
}

type StructWithComments struct {
	// First group of fields.

	// First is a field.
	First int // First field line comment.

	// Hidden group of fields.

	hidden int // Hidden field line comment.

	// Second group of fields.

	Second int

	// Embedded fields.

	*ExportedType

	// Trailing struct comment.
}
