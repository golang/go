package runtime

import (
	"internal/abi"
	"unsafe"
)

type GoLocalHolder[T any] struct {
	Val T
}

type _InnerGoLocalKey[T any] struct {
	rawKey any
	v0     *T
}

// newGoLocalObject creates a go_local object and record it.
func newGoLocalObject(key any, typ *_type) (pObject unsafe.Pointer, alloc bool) {
	gp := getg()
	ptr, ok := gp.localTable[key]
	if ok {
		return ptr, false
	}
	if gp.localTable == nil {
		gp.localTable = map[any]unsafe.Pointer{}
	}
	ptr = mallocgc(typ.Size_, typ, true)
	gp.localTable[key] = ptr
	return ptr, true
}

// newGoLocalObjectForStringKey wraps newGoLocalObject for ssa calling.
func newGoLocalObjectForStringKey(key string, typ *_type) (pObject unsafe.Pointer, alloc bool) {
	return newGoLocalObject(key, typ)
}

// NewGoLocal creates a go local object for rawKey + type and returns its holder.
// This can use the same one object in multiple places by the same rawKey + type
func NewGoLocal[T any](rawKey any, initFunc func() T) (ptrHolder *GoLocalHolder[T], alloc bool) {
	key := _InnerGoLocalKey[T]{rawKey: rawKey, v0: nil}
	wrapper0 := (*GoLocalHolder[T])(nil)
	ptr, alloc := newGoLocalObject(key, abi.TypeOf(wrapper0).Elem())
	ptrHolder = (*GoLocalHolder[T])(ptr)
	if alloc && initFunc != nil {
		ptrHolder.Val = initFunc()
	}
	return ptrHolder, alloc
}
