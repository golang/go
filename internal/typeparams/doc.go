// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package typeparams provides functions to work indirectly with type parameter
// data stored in go/ast and go/types objects, while these API are guarded by a
// build constraint.
//
// This package exists to make it easier for tools to work with generic code,
// while also compiling against older Go versions.
package typeparams
