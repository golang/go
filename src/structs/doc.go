// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package structs defines marker types that can be used as struct fields
// to modify the properties of a struct.
//
// By convention, a marker type should be used as the type of a field
// named "_", placed at the beginning of a struct type definition.
package structs
