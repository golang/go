// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/abi"
	"internal/runtime/maps"
	"internal/runtime/sys"
	"unsafe"
)

// Legacy //go:linkname compatibility shims
//
// The functions below are unused by the toolchain, and exist only for
// compatibility with existing //go:linkname use in the ecosystem.

// linknameIter is the it argument to mapiterinit and mapiternext.
//
// Callers of mapiterinit allocate their own iter structure, which has the
// layout of the pre-Go 1.24 hiter structure, shown here for posterity:
//
//	type hiter struct {
//		key         unsafe.Pointer
//		elem        unsafe.Pointer
//		t           *maptype // old map abi.Type
//		h           *hmap
//		buckets     unsafe.Pointer
//		bptr        *bmap
//		overflow    *[]*bmap
//		oldoverflow *[]*bmap
//		startBucket uintptr
//		offset      uint8
//		wrapped     bool
//		B           uint8
//		i           uint8
//		bucket      uintptr
//		checkBucket uintptr
//	}
//
// Our structure must maintain compatibility with the old structure. This
// means:
//
//   - Our structure must be the same size or smaller than hiter. Otherwise we
//     may write outside the caller's hiter allocation.
//   - Our structure must have the same pointer layout as hiter, so that the GC
//     tracks pointers properly.
//
// Based on analysis of the "hall of shame" users of these linknames:
//
//   - The key and elem fields must be kept up to date with the current key/elem.
//     Some users directly access the key and elem fields rather than calling
//     reflect.mapiterkey/reflect.mapiterelem.
//   - The t field must be non-nil after mapiterinit. gonum.org/v1/gonum uses
//     this to verify the iterator is initialized.
//   - github.com/segmentio/encoding and github.com/RomiChan/protobuf check if h
//     is non-nil, but the code has no effect. Thus the value of h does not
//     matter. See internal/runtime_reflect/map.go.
type linknameIter struct {
	// Fields from hiter.
	key  unsafe.Pointer
	elem unsafe.Pointer
	typ  *abi.MapType

	// The real iterator.
	it *maps.Iter
}

// mapiterinit is a compatibility wrapper for map iterator for users of
// //go:linkname from before Go 1.24. It is not used by Go itself. New users
// should use reflect or the maps package.
//
// mapiterinit should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/bytedance/sonic
//   - github.com/goccy/go-json
//   - github.com/RomiChan/protobuf
//   - github.com/segmentio/encoding
//   - github.com/ugorji/go/codec
//   - github.com/wI2L/jettison
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname mapiterinit
func mapiterinit(t *abi.MapType, m *maps.Map, it *linknameIter) {
	if raceenabled && m != nil {
		callerpc := sys.GetCallerPC()
		racereadpc(unsafe.Pointer(m), callerpc, abi.FuncPCABIInternal(mapiterinit))
	}

	it.typ = t

	it.it = new(maps.Iter)
	it.it.Init(t, m)
	it.it.Next()

	it.key = it.it.Key()
	it.elem = it.it.Elem()
}

// reflect_mapiterinit is a compatibility wrapper for map iterator for users of
// //go:linkname from before Go 1.24. It is not used by Go itself. New users
// should use reflect or the maps package.
//
// reflect_mapiterinit should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/modern-go/reflect2
//   - gitee.com/quant1x/gox
//   - github.com/v2pro/plz
//   - github.com/wI2L/jettison
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname reflect_mapiterinit reflect.mapiterinit
func reflect_mapiterinit(t *abi.MapType, m *maps.Map, it *linknameIter) {
	mapiterinit(t, m, it)
}

// mapiternext is a compatibility wrapper for map iterator for users of
// //go:linkname from before Go 1.24. It is not used by Go itself. New users
// should use reflect or the maps package.
//
// mapiternext should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/bytedance/sonic
//   - github.com/RomiChan/protobuf
//   - github.com/segmentio/encoding
//   - github.com/ugorji/go/codec
//   - gonum.org/v1/gonum
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname mapiternext
func mapiternext(it *linknameIter) {
	if raceenabled {
		callerpc := sys.GetCallerPC()
		racereadpc(unsafe.Pointer(it.it.Map()), callerpc, abi.FuncPCABIInternal(mapiternext))
	}

	it.it.Next()

	it.key = it.it.Key()
	it.elem = it.it.Elem()
}

// reflect_mapiternext is a compatibility wrapper for map iterator for users of
// //go:linkname from before Go 1.24. It is not used by Go itself. New users
// should use reflect or the maps package.
//
// reflect_mapiternext is for package reflect,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - gitee.com/quant1x/gox
//   - github.com/modern-go/reflect2
//   - github.com/goccy/go-json
//   - github.com/v2pro/plz
//   - github.com/wI2L/jettison
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname reflect_mapiternext reflect.mapiternext
func reflect_mapiternext(it *linknameIter) {
	mapiternext(it)
}

// reflect_mapiterkey is a compatibility wrapper for map iterator for users of
// //go:linkname from before Go 1.24. It is not used by Go itself. New users
// should use reflect or the maps package.
//
// reflect_mapiterkey should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/goccy/go-json
//   - gonum.org/v1/gonum
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname reflect_mapiterkey reflect.mapiterkey
func reflect_mapiterkey(it *linknameIter) unsafe.Pointer {
	return it.it.Key()
}

// reflect_mapiterelem is a compatibility wrapper for map iterator for users of
// //go:linkname from before Go 1.24. It is not used by Go itself. New users
// should use reflect or the maps package.
//
// reflect_mapiterelem should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/goccy/go-json
//   - gonum.org/v1/gonum
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname reflect_mapiterelem reflect.mapiterelem
func reflect_mapiterelem(it *linknameIter) unsafe.Pointer {
	return it.it.Elem()
}
