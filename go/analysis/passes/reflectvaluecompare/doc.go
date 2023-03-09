// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package reflectvaluecompare defines an Analyzer that checks for accidentally
// using == or reflect.DeepEqual to compare reflect.Value values.
// See issues 43993 and 18871.
//
// # Analyzer reflectvaluecompare
//
// reflectvaluecompare: check for comparing reflect.Value values with == or reflect.DeepEqual
//
// The reflectvaluecompare checker looks for expressions of the form:
//
//	v1 == v2
//	v1 != v2
//	reflect.DeepEqual(v1, v2)
//
// where v1 or v2 are reflect.Values. Comparing reflect.Values directly
// is almost certainly not correct, as it compares the reflect package's
// internal representation, not the underlying value.
// Likely what is intended is:
//
//	v1.Interface() == v2.Interface()
//	v1.Interface() != v2.Interface()
//	reflect.DeepEqual(v1.Interface(), v2.Interface())
package reflectvaluecompare
