// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// FieldDefault is a loGo/s struct field type with a default value
// (T-104.5): `Port int = 8080` parses the type and the `=` default as
// one node so shared declarations (`Lo, Hi int = 3`) keep one default.
// The checker unwraps it to Type; lowering strips the `= const` text
// and fills missing fields of composite literals with Default.
type FieldDefault struct {
	Type    Expr // declared field type
	Default Expr // default expression
	expr
}
