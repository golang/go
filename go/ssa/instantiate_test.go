// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// Note: Tests use unexported method _Instances.

import (
	"bytes"
	"fmt"
	"go/types"
	"reflect"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/internal/typeparams"
)

// loadProgram creates loader.Program out of p.
func loadProgram(p string) (*loader.Program, error) {
	// Parse
	var conf loader.Config
	f, err := conf.ParseFile("<input>", p)
	if err != nil {
		return nil, fmt.Errorf("parse: %v", err)
	}
	conf.CreateFromFiles("p", f)

	// Load
	lprog, err := conf.Load()
	if err != nil {
		return nil, fmt.Errorf("Load: %v", err)
	}
	return lprog, nil
}

// buildPackage builds and returns ssa representation of package pkg of lprog.
func buildPackage(lprog *loader.Program, pkg string, mode BuilderMode) *Package {
	prog := NewProgram(lprog.Fset, mode)

	for _, info := range lprog.AllPackages {
		prog.CreatePackage(info.Pkg, info.Files, &info.Info, info.Importable)
	}

	p := prog.Package(lprog.Package(pkg).Pkg)
	p.Build()
	return p
}

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

	lprog, err := loadProgram(input)
	if err != err {
		t.Fatal(err)
	}

	for _, mode := range []BuilderMode{BuilderMode(0), InstantiateGenerics} {
		// Create and build SSA
		p := buildPackage(lprog, "p", mode)
		prog := p.Prog

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
		if instance.Origin() != meth {
			t.Errorf("Expected Origin of %s to be %s. got %s", instance, meth, instance.Origin())
		}
		if len(instance.TypeArgs()) != 1 || !types.Identical(instance.TypeArgs()[0], intSliceTyp) {
			t.Errorf("Expected TypeArgs of %s to be %v. got %v", instance, []types.Type{intSliceTyp}, instance.typeargs)
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

// TestCallsToInstances checks that calles of calls to generic functions,
// without monomorphization, are wrappers around the origin generic function.
func TestCallsToInstances(t *testing.T) {
	if !typeparams.Enabled {
		return
	}
	const input = `
package p

type I interface {
	Foo()
}

type A int
func (a A) Foo() {}

type J[T any] interface{ Bar() T }
type K[T any] struct{ J[T] }

func Id[T any] (t T) T {
	return t
}

func Lambda[T I]() func() func(T) {
	return func() func(T) {
		return T.Foo
	}
}

func NoOp[T any]() {}

func Bar[T interface { Foo(); ~int | ~string }, U any] (t T, u U) {
	Id[U](u)
	Id[T](t)
}

func Make[T any]() interface{} {
	NoOp[K[T]]()
	return nil
}

func entry(i int, a A) int {
	Lambda[A]()()(a)

	x := Make[int]()
	if j, ok := x.(interface{ Bar() int }); ok {
		print(j)
	}

	Bar[A, int](a, i)

	return Id[int](i)
}
`
	lprog, err := loadProgram(input)
	if err != err {
		t.Fatal(err)
	}

	p := buildPackage(lprog, "p", SanityCheckFunctions)
	prog := p.Prog

	for _, ti := range []struct {
		orig         string
		instance     string
		tparams      string
		targs        string
		chTypeInstrs int // number of ChangeType instructions in f's body
	}{
		{"Id", "Id[int]", "[T]", "[int]", 2},
		{"Lambda", "Lambda[p.A]", "[T]", "[p.A]", 1},
		{"Make", "Make[int]", "[T]", "[int]", 0},
		{"NoOp", "NoOp[p.K[T]]", "[T]", "[p.K[T]]", 0},
	} {
		test := ti
		t.Run(test.instance, func(t *testing.T) {
			f := p.Members[test.orig].(*Function)
			if f == nil {
				t.Fatalf("origin function not found")
			}

			i := instanceOf(f, test.instance, prog)
			if i == nil {
				t.Fatalf("instance not found")
			}

			// for logging on failures
			var body strings.Builder
			i.WriteTo(&body)
			t.Log(body.String())

			if len(i.Blocks) != 1 {
				t.Fatalf("body has more than 1 block")
			}

			if instrs := changeTypeInstrs(i.Blocks[0]); instrs != test.chTypeInstrs {
				t.Errorf("want %v instructions; got %v", test.chTypeInstrs, instrs)
			}

			if test.tparams != tparams(i) {
				t.Errorf("want %v type params; got %v", test.tparams, tparams(i))
			}

			if test.targs != targs(i) {
				t.Errorf("want %v type arguments; got %v", test.targs, targs(i))
			}
		})
	}
}

func instanceOf(f *Function, name string, prog *Program) *Function {
	for _, i := range prog._Instances(f) {
		if i.Name() == name {
			return i
		}
	}
	return nil
}

func tparams(f *Function) string {
	tplist := f.TypeParams()
	var tps []string
	for i := 0; i < tplist.Len(); i++ {
		tps = append(tps, tplist.At(i).String())
	}
	return fmt.Sprint(tps)
}

func targs(f *Function) string {
	var tas []string
	for _, ta := range f.TypeArgs() {
		tas = append(tas, ta.String())
	}
	return fmt.Sprint(tas)
}

func changeTypeInstrs(b *BasicBlock) int {
	cnt := 0
	for _, i := range b.Instrs {
		if _, ok := i.(*ChangeType); ok {
			cnt++
		}
	}
	return cnt
}

func TestInstanceUniqueness(t *testing.T) {
	if !typeparams.Enabled {
		return
	}
	const input = `
package p

func H[T any](t T) {
	print(t)
}

func F[T any](t T) {
	H[T](t)
	H[T](t)
	H[T](t)
}

func G[T any](t T) {
	H[T](t)
	H[T](t)
}

func Foo[T any, S any](t T, s S) {
	Foo[S, T](s, t)
	Foo[T, S](t, s)
}
`
	lprog, err := loadProgram(input)
	if err != err {
		t.Fatal(err)
	}

	p := buildPackage(lprog, "p", SanityCheckFunctions)
	prog := p.Prog

	for _, test := range []struct {
		orig      string
		instances string
	}{
		{"H", "[p.H[T] p.H[T]]"},
		{"Foo", "[p.Foo[S T] p.Foo[T S]]"},
	} {
		t.Run(test.orig, func(t *testing.T) {
			f := p.Members[test.orig].(*Function)
			if f == nil {
				t.Fatalf("origin function not found")
			}

			instances := prog._Instances(f)
			sort.Slice(instances, func(i, j int) bool { return instances[i].Name() < instances[j].Name() })

			if got := fmt.Sprintf("%v", instances); !reflect.DeepEqual(got, test.instances) {
				t.Errorf("got %v instances, want %v", got, test.instances)
			}
		})
	}
}

// instancesStr returns a sorted slice of string
// representation of instances.
func instancesStr(instances []*Function) []string {
	var is []string
	for _, i := range instances {
		is = append(is, fmt.Sprintf("%v", i))
	}
	sort.Strings(is)
	return is
}
