// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package structs

// HostLayout marks a struct as using host memory layout. A struct with a
// field of type HostLayout will be laid out in memory according to host
// expectations, generally following the host's C ABI.
//
// HostLayout does not affect layout within any other struct-typed fields
// of the containing struct, nor does it affect layout of structs
// containing the struct marked as host layout.
//
// By convention, HostLayout should be used as the type of a field
// named "_", placed at the beginning of the struct type definition.
type HostLayout struct {
	_ hostLayout // prevent accidental conversion with plain struct{}
}

// We use an unexported type within the exported type to give the marker
// type itself, rather than merely its name, a recognizable identity in
// the type system. The main consequence of this is that a user can give
// the type a new name and it will still have the same properties, e.g.,
//
//	type HL structs.HostLayout
//
// It also prevents unintentional conversion of struct{} to a named marker type.
type hostLayout struct {
}
