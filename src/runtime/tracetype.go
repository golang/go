// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Trace stack table and acquisition.

package runtime

import (
	"internal/abi"
	"internal/goarch"
	"internal/trace/tracev2"
	"unsafe"
)

// traceTypeTable maps stack traces (arrays of PC's) to unique uint32 ids.
// It is lock-free for reading.
type traceTypeTable struct {
	tab traceMap
}

// put returns a unique id for the type typ and caches it in the table,
// if it's seeing it for the first time.
//
// N.B. typ must be kept alive forever for this to work correctly.
func (t *traceTypeTable) put(typ *abi.Type) uint64 {
	if typ == nil {
		return 0
	}
	// Insert the pointer to the type itself.
	id, _ := t.tab.put(noescape(unsafe.Pointer(&typ)), goarch.PtrSize)
	return id
}

// dump writes all previously cached types to trace buffers and
// releases all memory and resets state. It must only be called once the caller
// can guarantee that there are no more writers to the table.
func (t *traceTypeTable) dump(gen uintptr) {
	w := unsafeTraceExpWriter(gen, nil, tracev2.AllocFree)
	if root := (*traceMapNode)(t.tab.root.Load()); root != nil {
		w = dumpTypesRec(root, w)
	}
	w.flush().end()
	t.tab.reset()
}

func dumpTypesRec(node *traceMapNode, w traceWriter) traceWriter {
	typ := (*abi.Type)(*(*unsafe.Pointer)(unsafe.Pointer(&node.data[0])))
	typName := toRType(typ).string()

	// The maximum number of bytes required to hold the encoded type.
	maxBytes := 1 + 5*traceBytesPerNumber + len(typName)

	// Estimate the size of this record. This
	// bound is pretty loose, but avoids counting
	// lots of varint sizes.
	//
	// Add 1 because we might also write a traceAllocFreeTypesBatch byte.
	var flushed bool
	w, flushed = w.ensure(1 + maxBytes)
	if flushed {
		// Annotate the batch as containing types.
		w.byte(byte(traceAllocFreeTypesBatch))
	}

	// Emit type.
	w.varint(node.id)
	w.varint(uint64(uintptr(unsafe.Pointer(typ))))
	w.varint(uint64(typ.Size()))
	w.varint(uint64(typ.PtrBytes))
	w.varint(uint64(len(typName)))
	w.stringData(typName)

	// Recursively walk all child nodes.
	for i := range node.children {
		child := node.children[i].Load()
		if child == nil {
			continue
		}
		w = dumpTypesRec((*traceMapNode)(child), w)
	}
	return w
}
