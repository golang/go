// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// Note: Tests use unexported functions.

import (
	"bytes"
	"go/types"
	"reflect"
	"sort"
	"testing"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/internal/typeparams"
)

// TestNeedsInstance ensures that new method instances can be created via needsInstance,
// that TypeArgs are as expected, and can be accessed via _Instances.
func TestNeedsInstance(t *testing.T) {
	if !typeparams.Enabled {
		return
	}
	const input = `
package p

import "unsafe"

type Pointer[T any] struct {
	v unsafe.Pointer
}

func (x *Pointer[T]) Load() *T {
	return (*T)(LoadPointer(&x.v))
}

func LoadPointer(addr *unsafe.Pointer) (val unsafe.Pointer)
`
	// The SSA members for this package should look something like this:
	//      func  LoadPointer func(addr *unsafe.Pointer) (val unsafe.Pointer)
	//      type  Pointer     struct{v unsafe.Pointer}
	//        method (*Pointer[T any]) Load() *T
	//      func  init        func()
	//      var   init$guard  bool

	// Parse
	var conf loader.Config
	f, err := conf.ParseFile("<input>", input)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	conf.CreateFromFiles("p", f)

	// Load
	lprog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	for _, mode := range []BuilderMode{BuilderMode(0), InstantiateGenerics} {
		// Create and build SSA
		prog := NewProgram(lprog.Fset, mode)

		for _, info := range lprog.AllPackages {
			prog.CreatePackage(info.Pkg, info.Files, &info.Info, info.Importable)
		}

		p := prog.Package(lprog.Package("p").Pkg)
		p.Build()

		ptr := p.Type("Pointer").Type().(*types.Named)
		if ptr.NumMethods() != 1 {
			t.Fatalf("Expected Pointer to have 1 method. got %d", ptr.NumMethods())
		}

		obj := ptr.Method(0)
		if obj.Name() != "Load" {
			t.Errorf("Expected Pointer to have method named 'Load'. got %q", obj.Name())
		}

		meth := prog.FuncValue(obj)

		var cr creator
		intSliceTyp := types.NewSlice(types.Typ[types.Int])
		instance := prog.needsInstance(meth, []types.Type{intSliceTyp}, &cr)
		if len(cr) != 1 {
			t.Errorf("Expected first instance to create a function. got %d created functions", len(cr))
		}
		if instance._Origin != meth {
			t.Errorf("Expected Origin of %s to be %s. got %s", instance, meth, instance._Origin)
		}
		if len(instance._TypeArgs) != 1 || !types.Identical(instance._TypeArgs[0], intSliceTyp) {
			t.Errorf("Expected TypeArgs of %s to be %v. got %v", instance, []types.Type{intSliceTyp}, instance._TypeArgs)
		}
		instances := prog._Instances(meth)
		if want := []*Function{instance}; !reflect.DeepEqual(instances, want) {
			t.Errorf("Expected instances of %s to be %v. got %v", meth, want, instances)
		}

		// A second request with an identical type returns the same Function.
		second := prog.needsInstance(meth, []types.Type{types.NewSlice(types.Typ[types.Int])}, &cr)
		if second != instance || len(cr) != 1 {
			t.Error("Expected second identical instantiation to not create a function")
		}

		// Add a second instance.
		inst2 := prog.needsInstance(meth, []types.Type{types.NewSlice(types.Typ[types.Uint])}, &cr)
		instances = prog._Instances(meth)

		// Note: instance.Name() < inst2.Name()
		sort.Slice(instances, func(i, j int) bool {
			return instances[i].Name() < instances[j].Name()
		})
		if want := []*Function{instance, inst2}; !reflect.DeepEqual(instances, want) {
			t.Errorf("Expected instances of %s to be %v. got %v", meth, want, instances)
		}

		// build and sanity check manually created instance.
		var b builder
		b.buildFunction(instance)
		var buf bytes.Buffer
		if !sanityCheck(instance, &buf) {
			t.Errorf("sanityCheck of %s failed with: %s", instance, buf.String())
		}
	}
}
