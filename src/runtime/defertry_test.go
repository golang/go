// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"
	"time"
	"unsafe"
)

func deferTryCtype(t *testing.T, v any) unsafe.Pointer {
	t.Helper()
	return unsafe.Pointer(runtime.DeferTryTypeOf(v))
}

func deferTryFields(words int, ptrAt []int) []reflect.StructField {
	fields := make([]reflect.StructField, 0, words+1)
	fields = append(fields, reflect.StructField{Name: "R", Type: reflect.TypeOf(uintptr(0))})
	isPtr := func(i int) bool {
		for _, p := range ptrAt {
			if p == i {
				return true
			}
		}
		return false
	}
	need := words - 1
	for i := 0; i < need; {
		name := fmt.Sprintf("F%d", i)
		if isPtr(i) {
			fields = append(fields, reflect.StructField{Name: name, Type: reflect.TypeOf((*int)(nil))})
			i++
			continue
		}
		if i+1 < need && !isPtr(i+1) && len(fields)%2 == 0 {
			fields = append(fields, reflect.StructField{Name: name, Type: reflect.TypeOf([2]uintptr{})})
			i += 2
			continue
		}
		fields = append(fields, reflect.StructField{Name: name, Type: reflect.TypeOf(uintptr(0))})
		i++
	}
	return fields
}

func deferTryBuild(t *testing.T, fields []reflect.StructField, stackErr, heapErr *error) (reflect.Value, uintptr) {
	t.Helper()
	typ := reflect.StructOf(fields)
	rv := reflect.New(typ)
	re := rv.Elem()
	for i := range fields {
		f := fields[i]
		if f.Name == "R" {
			continue
		}
		fv := re.Field(i)
		switch f.Type.Kind() {
		case reflect.Ptr:
			p := new(int)
			*p = 1000 + i
			fv.Set(reflect.ValueOf(p))
		case reflect.Uintptr:
			fv.SetUint(uint64(2000 + i))
		case reflect.Array:
			for j := 0; j < fv.Len(); j++ {
				fv.Index(j).SetUint(uint64(3000 + i*10 + j))
			}
		}
		_ = heapErr
	}
	re.FieldByName("R").SetUint(uint64(uintptr(unsafe.Pointer(stackErr))))
	off, ok := typ.FieldByName("R")
	if !ok {
		t.Fatal("no R field")
	}
	return rv, off.Offset
}

func TestDeferTryRoundTrip(t *testing.T) {
	var stackErr error
	heapErr := new(error)
	*heapErr = nil
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 25, 40}
	for _, n := range sizes {
		for _, mode := range []int{0, 1, 2} {
			var ptrs []int
			switch mode {
			case 1:
				ptrs = []int{n - 1}
			case 2:
				for i := 0; i < n; i += 2 {
					ptrs = append(ptrs, i)
				}
			}
			fields := deferTryFields(n, ptrs)
			rv, roff := deferTryBuild(t, fields, &stackErr, heapErr)
			ctype := deferTryCtype(t, rv.Elem().Interface())
			rec := runtime.DeferTryNew()
			runtime.DeferTryPack(rec, ctype, rv.UnsafePointer(), roff)
			dst := reflect.New(rv.Elem().Type())
			runtime.DeferTryUnpackRec(runtime.DeferTryRec(rec), ctype, dst.UnsafePointer(), roff)
			if !reflect.DeepEqual(rv.Elem().Interface(), dst.Elem().Interface()) {
				t.Fatalf("size %d mode %d mismatch", n, mode)
			}
			if *(*unsafe.Pointer)(unsafe.Pointer(uintptr(dst.UnsafePointer()) + roff)) != unsafe.Pointer(&stackErr) {
				t.Fatalf("size %d mode %d err slot not stack", n, mode)
			}
		}
	}
	fields := deferTryFields(6, nil)
	rv, roff := deferTryBuild(t, fields, heapErr, heapErr)
	ctype := deferTryCtype(t, rv.Elem().Interface())
	rec := runtime.DeferTryNew()
	runtime.DeferTryPack(rec, ctype, rv.UnsafePointer(), roff)
	dst := reflect.New(rv.Elem().Type())
	runtime.DeferTryUnpackRec(runtime.DeferTryRec(rec), ctype, dst.UnsafePointer(), roff)
	if !reflect.DeepEqual(rv.Elem().Interface(), dst.Elem().Interface()) {
		t.Fatal("heap err slot mismatch")
	}
	if *(*unsafe.Pointer)(unsafe.Pointer(uintptr(dst.UnsafePointer()) + roff)) != unsafe.Pointer(heapErr) {
		t.Fatal("err slot not heap")
	}
}

func TestDeferTryGC(t *testing.T) {
	var stackErr error
	type pair struct {
		R uintptr
		P *int
		Q *int
		S uintptr
	}
	fin := make(chan int, 2)
	mk := func(v int) *int {
		p := new(int)
		*p = v
		runtime.AddCleanup(p, func(v int) { fin <- v }, v)
		return p
	}
	mkPair := func() (any, uintptr) {
		v := pair{R: uintptr(unsafe.Pointer(&stackErr)), P: mk(1), Q: mk(2), S: 7}
		typ := reflect.TypeOf(v)
		f, _ := typ.FieldByName("R")
		return v, f.Offset
	}
	v, roff := mkPair()
	ctype := deferTryCtype(t, v)
	rv := reflect.New(reflect.TypeOf(v))
	rv.Elem().Set(reflect.ValueOf(v))
	rec := runtime.DeferTryNew()
	runtime.DeferTryPack(rec, ctype, rv.UnsafePointer(), roff)
	rv = reflect.Value{}
	v = nil
	runtime.GC()
	runtime.GC()
	sacc := make(chan int, 1)
	runtime.AddCleanup(new(int), func(v int) { sacc <- v }, 3)
	deadline := time.Now().Add(30 * time.Second)
	cycled := false
	for !cycled {
		runtime.GC()
		select {
		case <-sacc:
			cycled = true
		default:
			runtime.Gosched()
		}
		if time.Now().After(deadline) {
			t.Fatal("cleanup worker did not cycle")
		}
	}
	select {
	case got := <-fin:
		t.Fatalf("cleanup ran while record live: %d", got)
	default:
	}
	dst := reflect.New(reflect.TypeOf(pair{}))
	runtime.DeferTryUnpackRec(runtime.DeferTryRec(rec), ctype, dst.UnsafePointer(), roff)
	got := dst.Elem().Interface().(pair)
	if got.S != 7 || got.P == nil || *got.P != 1 || got.Q == nil || *got.Q != 2 {
		t.Fatalf("unpack mismatch: %+v", got)
	}
	got.P, got.Q = nil, nil
	dst = reflect.Value{}
	runtime.GC()
	deadline = time.Now().Add(30 * time.Second)
	for len(fin) != 2 {
		if time.Now().After(deadline) {
			t.Fatalf("want 2 cleanups after free, got %d", len(fin))
		}
		runtime.GC()
		runtime.Gosched()
	}
}

func TestDeferTryErroff(t *testing.T) {
	var stackErr error
	fields := deferTryFields(2, nil)
	rv, roff := deferTryBuild(t, fields, &stackErr, nil)
	want := uintptr(unsafe.Pointer(&stackErr))
	if *(*uintptr)(rv.UnsafePointer()) != want {
		t.Fatal("setup did not point R at the error result")
	}
	ctype := deferTryCtype(t, rv.Elem().Interface())
	typ := rv.Elem().Type()
	rec := runtime.DeferTryNew()
	runtime.DeferTryPack(rec, ctype, rv.UnsafePointer(), roff)
	rv = reflect.Value{}
	hi0 := runtime.DeferTryStackHi()
	var got uintptr
	var grow func(d int)
	grow = func(d int) {
		var pad [1024]byte
		for i := range pad {
			pad[i] = byte(d + i)
		}
		if d == 0 {
			dst := reflect.New(typ)
			runtime.DeferTryUnpackRec(runtime.DeferTryRec(rec), ctype, dst.UnsafePointer(), roff)
			got = *(*uintptr)(dst.UnsafePointer())
			return
		}
		grow(d - 1)
	}
	grow(512)
	if runtime.DeferTryStackHi() == hi0 {
		t.Fatal("stack did not grow")
	}
	if got != uintptr(unsafe.Pointer(&stackErr)) {
		t.Fatalf("resolved err slot %#x, want %#x", got, uintptr(unsafe.Pointer(&stackErr)))
	}
}

func TestDeferTryPoolSteady(t *testing.T) {
	var stackErr error
	fields := deferTryFields(9, []int{0, 2, 4})
	rv, roff := deferTryBuild(t, fields, &stackErr, nil)
	rv.Elem().FieldByName("R").SetUint(uint64(uintptr(unsafe.Pointer(&stackErr))))
	ctype := deferTryCtype(t, rv.Elem().Interface())
	dst := reflect.New(rv.Elem().Type())
	n := testing.AllocsPerRun(100, func() {
		rec := runtime.DeferTryNew()
		runtime.DeferTryPack(rec, ctype, rv.UnsafePointer(), roff)
		runtime.DeferTryUnpackRec(runtime.DeferTryRec(rec), ctype, dst.UnsafePointer(), roff)
	})
	if n != 0 {
		t.Fatalf("pack/unpack/free allocates %v per run, want 0", n)
	}
}

func TestDeferTryFreeZeroesExt(t *testing.T) {
	var stackErr error
	fin := make(chan int, 1)
	mk := func(v int) *int {
		p := new(int)
		*p = v
		runtime.AddCleanup(p, func(v int) { fin <- v }, v)
		return p
	}
	fields := deferTryFields(25, []int{0, 1, 2, 3, 4, 20})
	pname := ""
	for i := len(fields) - 1; i >= 0; i-- {
		if fields[i].Type.Kind() == reflect.Ptr {
			pname = fields[i].Name
			break
		}
	}
	if pname == "" {
		t.Fatal("no pointer field")
	}
	rv, roff := deferTryBuild(t, fields, &stackErr, nil)
	rv.Elem().FieldByName(pname).Set(reflect.ValueOf(mk(9)))
	ctype := deferTryCtype(t, rv.Elem().Interface())
	rec := runtime.DeferTryNew()
	runtime.DeferTryPack(rec, ctype, rv.UnsafePointer(), roff)
	rv = reflect.Value{}
	runtime.DeferTryFree(rec)
	deadline := time.Now().Add(30 * time.Second)
	for len(fin) == 0 {
		if time.Now().After(deadline) {
			t.Fatal("free did not release ext referent")
		}
		runtime.GC()
		runtime.Gosched()
	}
	rv2, _ := deferTryBuild(t, fields, &stackErr, nil)
	typ2 := rv2.Elem().Type()
	for range 8 {
		rec := runtime.DeferTryNew()
		runtime.DeferTryPack(rec, ctype, rv2.UnsafePointer(), roff)
		dst := reflect.New(typ2)
		runtime.DeferTryUnpackRec(runtime.DeferTryRec(rec), ctype, dst.UnsafePointer(), roff)
		if !reflect.DeepEqual(rv2.Elem().Interface(), dst.Elem().Interface()) {
			t.Fatal("reuse mismatch")
		}
	}
}
