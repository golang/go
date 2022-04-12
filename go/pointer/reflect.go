// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

// This file implements the generation and resolution rules for
// constraints arising from the use of reflection in the target
// program.  See doc.go for explanation of the representation.
//
// For consistency, the names of all parameters match those of the
// actual functions in the "reflect" package.
//
// To avoid proliferation of equivalent labels, intrinsics should
// memoize as much as possible, like TypeOf and Zero do for their
// tagged objects.
//
// TODO(adonovan): this file is rather subtle.  Explain how we derive
// the implementation of each reflect operator from its spec,
// including the subtleties of reflect.flag{Addr,RO,Indir}.
// [Hint: our implementation is as if reflect.flagIndir was always
// true, i.e. reflect.Values are pointers to tagged objects, there is
// no inline allocation optimization; and indirect tagged objects (not
// yet implemented) correspond to reflect.Values with
// reflect.flagAddr.]
// A picture would help too.
//
// TODO(adonovan): try factoring up the common parts of the majority of
// these constraints that are single input, single output.

import (
	"fmt"
	"go/constant"
	"go/types"
	"reflect"

	"golang.org/x/tools/go/ssa"
)

func init() {
	for name, fn := range map[string]intrinsic{
		// reflect.Value methods.
		"(reflect.Value).Addr":            ext۰reflect۰Value۰Addr,
		"(reflect.Value).Bool":            ext۰NoEffect,
		"(reflect.Value).Bytes":           ext۰reflect۰Value۰Bytes,
		"(reflect.Value).Call":            ext۰reflect۰Value۰Call,
		"(reflect.Value).CallSlice":       ext۰reflect۰Value۰CallSlice,
		"(reflect.Value).CanAddr":         ext۰NoEffect,
		"(reflect.Value).CanInterface":    ext۰NoEffect,
		"(reflect.Value).CanSet":          ext۰NoEffect,
		"(reflect.Value).Cap":             ext۰NoEffect,
		"(reflect.Value).Close":           ext۰NoEffect,
		"(reflect.Value).Complex":         ext۰NoEffect,
		"(reflect.Value).Convert":         ext۰reflect۰Value۰Convert,
		"(reflect.Value).Elem":            ext۰reflect۰Value۰Elem,
		"(reflect.Value).Field":           ext۰reflect۰Value۰Field,
		"(reflect.Value).FieldByIndex":    ext۰reflect۰Value۰FieldByIndex,
		"(reflect.Value).FieldByName":     ext۰reflect۰Value۰FieldByName,
		"(reflect.Value).FieldByNameFunc": ext۰reflect۰Value۰FieldByNameFunc,
		"(reflect.Value).Float":           ext۰NoEffect,
		"(reflect.Value).Index":           ext۰reflect۰Value۰Index,
		"(reflect.Value).Int":             ext۰NoEffect,
		"(reflect.Value).Interface":       ext۰reflect۰Value۰Interface,
		"(reflect.Value).InterfaceData":   ext۰NoEffect,
		"(reflect.Value).IsNil":           ext۰NoEffect,
		"(reflect.Value).IsValid":         ext۰NoEffect,
		"(reflect.Value).Kind":            ext۰NoEffect,
		"(reflect.Value).Len":             ext۰NoEffect,
		"(reflect.Value).MapIndex":        ext۰reflect۰Value۰MapIndex,
		"(reflect.Value).MapKeys":         ext۰reflect۰Value۰MapKeys,
		"(reflect.Value).Method":          ext۰reflect۰Value۰Method,
		"(reflect.Value).MethodByName":    ext۰reflect۰Value۰MethodByName,
		"(reflect.Value).NumField":        ext۰NoEffect,
		"(reflect.Value).NumMethod":       ext۰NoEffect,
		"(reflect.Value).OverflowComplex": ext۰NoEffect,
		"(reflect.Value).OverflowFloat":   ext۰NoEffect,
		"(reflect.Value).OverflowInt":     ext۰NoEffect,
		"(reflect.Value).OverflowUint":    ext۰NoEffect,
		"(reflect.Value).Pointer":         ext۰NoEffect,
		"(reflect.Value).Recv":            ext۰reflect۰Value۰Recv,
		"(reflect.Value).Send":            ext۰reflect۰Value۰Send,
		"(reflect.Value).Set":             ext۰reflect۰Value۰Set,
		"(reflect.Value).SetBool":         ext۰NoEffect,
		"(reflect.Value).SetBytes":        ext۰reflect۰Value۰SetBytes,
		"(reflect.Value).SetComplex":      ext۰NoEffect,
		"(reflect.Value).SetFloat":        ext۰NoEffect,
		"(reflect.Value).SetInt":          ext۰NoEffect,
		"(reflect.Value).SetLen":          ext۰NoEffect,
		"(reflect.Value).SetMapIndex":     ext۰reflect۰Value۰SetMapIndex,
		"(reflect.Value).SetPointer":      ext۰reflect۰Value۰SetPointer,
		"(reflect.Value).SetString":       ext۰NoEffect,
		"(reflect.Value).SetUint":         ext۰NoEffect,
		"(reflect.Value).Slice":           ext۰reflect۰Value۰Slice,
		"(reflect.Value).String":          ext۰NoEffect,
		"(reflect.Value).TryRecv":         ext۰reflect۰Value۰Recv,
		"(reflect.Value).TrySend":         ext۰reflect۰Value۰Send,
		"(reflect.Value).Type":            ext۰NoEffect,
		"(reflect.Value).Uint":            ext۰NoEffect,
		"(reflect.Value).UnsafeAddr":      ext۰NoEffect,

		// Standalone reflect.* functions.
		"reflect.Append":      ext۰reflect۰Append,
		"reflect.AppendSlice": ext۰reflect۰AppendSlice,
		"reflect.Copy":        ext۰reflect۰Copy,
		"reflect.ChanOf":      ext۰reflect۰ChanOf,
		"reflect.DeepEqual":   ext۰NoEffect,
		"reflect.Indirect":    ext۰reflect۰Indirect,
		"reflect.MakeChan":    ext۰reflect۰MakeChan,
		"reflect.MakeFunc":    ext۰reflect۰MakeFunc,
		"reflect.MakeMap":     ext۰reflect۰MakeMap,
		"reflect.MakeSlice":   ext۰reflect۰MakeSlice,
		"reflect.MapOf":       ext۰reflect۰MapOf,
		"reflect.New":         ext۰reflect۰New,
		"reflect.NewAt":       ext۰reflect۰NewAt,
		"reflect.PtrTo":       ext۰reflect۰PtrTo,
		"reflect.Select":      ext۰reflect۰Select,
		"reflect.SliceOf":     ext۰reflect۰SliceOf,
		"reflect.TypeOf":      ext۰reflect۰TypeOf,
		"reflect.ValueOf":     ext۰reflect۰ValueOf,
		"reflect.Zero":        ext۰reflect۰Zero,
		"reflect.init":        ext۰NoEffect,

		// *reflect.rtype methods
		"(*reflect.rtype).Align":           ext۰NoEffect,
		"(*reflect.rtype).AssignableTo":    ext۰NoEffect,
		"(*reflect.rtype).Bits":            ext۰NoEffect,
		"(*reflect.rtype).ChanDir":         ext۰NoEffect,
		"(*reflect.rtype).ConvertibleTo":   ext۰NoEffect,
		"(*reflect.rtype).Elem":            ext۰reflect۰rtype۰Elem,
		"(*reflect.rtype).Field":           ext۰reflect۰rtype۰Field,
		"(*reflect.rtype).FieldAlign":      ext۰NoEffect,
		"(*reflect.rtype).FieldByIndex":    ext۰reflect۰rtype۰FieldByIndex,
		"(*reflect.rtype).FieldByName":     ext۰reflect۰rtype۰FieldByName,
		"(*reflect.rtype).FieldByNameFunc": ext۰reflect۰rtype۰FieldByNameFunc,
		"(*reflect.rtype).Implements":      ext۰NoEffect,
		"(*reflect.rtype).In":              ext۰reflect۰rtype۰In,
		"(*reflect.rtype).IsVariadic":      ext۰NoEffect,
		"(*reflect.rtype).Key":             ext۰reflect۰rtype۰Key,
		"(*reflect.rtype).Kind":            ext۰NoEffect,
		"(*reflect.rtype).Len":             ext۰NoEffect,
		"(*reflect.rtype).Method":          ext۰reflect۰rtype۰Method,
		"(*reflect.rtype).MethodByName":    ext۰reflect۰rtype۰MethodByName,
		"(*reflect.rtype).Name":            ext۰NoEffect,
		"(*reflect.rtype).NumField":        ext۰NoEffect,
		"(*reflect.rtype).NumIn":           ext۰NoEffect,
		"(*reflect.rtype).NumMethod":       ext۰NoEffect,
		"(*reflect.rtype).NumOut":          ext۰NoEffect,
		"(*reflect.rtype).Out":             ext۰reflect۰rtype۰Out,
		"(*reflect.rtype).PkgPath":         ext۰NoEffect,
		"(*reflect.rtype).Size":            ext۰NoEffect,
		"(*reflect.rtype).String":          ext۰NoEffect,
	} {
		intrinsicsByName[name] = fn
	}
}

// -------------------- (reflect.Value) --------------------

func ext۰reflect۰Value۰Addr(a *analysis, cgn *cgnode) {} // TODO(adonovan)

// ---------- func (Value).Bytes() Value ----------

// result = v.Bytes()
type rVBytesConstraint struct {
	v      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *rVBytesConstraint) ptr() nodeid { return c.v }
func (c *rVBytesConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "rVBytes.result")
}
func (c *rVBytesConstraint) renumber(mapping []nodeid) {
	c.v = mapping[c.v]
	c.result = mapping[c.result]
}

func (c *rVBytesConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.Bytes()", c.result, c.v)
}

func (c *rVBytesConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		vObj := nodeid(x)
		tDyn, slice, indirect := a.taggedValue(vObj)
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		tSlice, ok := tDyn.Underlying().(*types.Slice)
		if ok && types.Identical(tSlice.Elem(), types.Typ[types.Uint8]) {
			if a.onlineCopy(c.result, slice) {
				changed = true
			}
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰Value۰Bytes(a *analysis, cgn *cgnode) {
	a.addConstraint(&rVBytesConstraint{
		v:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func (Value).Call(in []Value) []Value ----------

// result = v.Call(in)
type rVCallConstraint struct {
	cgn       *cgnode
	targets   nodeid // (indirect)
	v         nodeid // (ptr)
	arg       nodeid // = in[*]
	result    nodeid // (indirect)
	dotdotdot bool   // interpret last arg as a "..." slice
}

func (c *rVCallConstraint) ptr() nodeid { return c.v }
func (c *rVCallConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.targets), "rVCall.targets")
	h.markIndirect(onodeid(c.result), "rVCall.result")
}
func (c *rVCallConstraint) renumber(mapping []nodeid) {
	c.targets = mapping[c.targets]
	c.v = mapping[c.v]
	c.arg = mapping[c.arg]
	c.result = mapping[c.result]
}

func (c *rVCallConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.Call(n%d)", c.result, c.v, c.arg)
}

func (c *rVCallConstraint) solve(a *analysis, delta *nodeset) {
	if c.targets == 0 {
		panic("no targets")
	}

	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		vObj := nodeid(x)
		tDyn, fn, indirect := a.taggedValue(vObj)
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		tSig, ok := tDyn.Underlying().(*types.Signature)
		if !ok {
			continue // not a function
		}
		if tSig.Recv() != nil {
			panic(tSig) // TODO(adonovan): rethink when we implement Method()
		}

		// Add dynamic call target.
		if a.onlineCopy(c.targets, fn) {
			a.addWork(c.targets)
			// TODO(adonovan): is 'else continue' a sound optimisation here?
		}

		// Allocate a P/R block.
		tParams := tSig.Params()
		tResults := tSig.Results()
		params := a.addNodes(tParams, "rVCall.params")
		results := a.addNodes(tResults, "rVCall.results")

		// Make a dynamic call to 'fn'.
		a.store(fn, params, 1, a.sizeof(tParams))
		a.load(results, fn, 1+a.sizeof(tParams), a.sizeof(tResults))

		// Populate P by type-asserting each actual arg (all merged in c.arg).
		for i, n := 0, tParams.Len(); i < n; i++ {
			T := tParams.At(i).Type()
			a.typeAssert(T, params, c.arg, false)
			params += nodeid(a.sizeof(T))
		}

		// Use R by tagging and copying each actual result to c.result.
		for i, n := 0, tResults.Len(); i < n; i++ {
			T := tResults.At(i).Type()
			// Convert from an arbitrary type to a reflect.Value
			// (like MakeInterface followed by reflect.ValueOf).
			if isInterface(T) {
				// (don't tag)
				if a.onlineCopy(c.result, results) {
					changed = true
				}
			} else {
				obj := a.makeTagged(T, c.cgn, nil)
				a.onlineCopyN(obj+1, results, a.sizeof(T))
				if a.addLabel(c.result, obj) { // (true)
					changed = true
				}
			}
			results += nodeid(a.sizeof(T))
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

// Common code for direct (inlined) and indirect calls to (reflect.Value).Call.
func reflectCallImpl(a *analysis, cgn *cgnode, site *callsite, recv, arg nodeid, dotdotdot bool) nodeid {
	// Allocate []reflect.Value array for the result.
	ret := a.nextNode()
	a.addNodes(types.NewArray(a.reflectValueObj.Type(), 1), "rVCall.ret")
	a.endObject(ret, cgn, nil)

	// pts(targets) will be the set of possible call targets.
	site.targets = a.addOneNode(tInvalid, "rvCall.targets", nil)

	// All arguments are merged since they arrive in a slice.
	argelts := a.addOneNode(a.reflectValueObj.Type(), "rVCall.args", nil)
	a.load(argelts, arg, 1, 1) // slice elements

	a.addConstraint(&rVCallConstraint{
		cgn:       cgn,
		targets:   site.targets,
		v:         recv,
		arg:       argelts,
		result:    ret + 1, // results go into elements of ret
		dotdotdot: dotdotdot,
	})
	return ret
}

func reflectCall(a *analysis, cgn *cgnode, dotdotdot bool) {
	// This is the shared contour implementation of (reflect.Value).Call
	// and CallSlice, as used by indirect calls (rare).
	// Direct calls are inlined in gen.go, eliding the
	// intermediate cgnode for Call.
	site := new(callsite)
	cgn.sites = append(cgn.sites, site)
	recv := a.funcParams(cgn.obj)
	arg := recv + 1
	ret := reflectCallImpl(a, cgn, site, recv, arg, dotdotdot)
	a.addressOf(cgn.fn.Signature.Results().At(0).Type(), a.funcResults(cgn.obj), ret)
}

func ext۰reflect۰Value۰Call(a *analysis, cgn *cgnode) {
	reflectCall(a, cgn, false)
}

func ext۰reflect۰Value۰CallSlice(a *analysis, cgn *cgnode) {
	// TODO(adonovan): implement.  Also, inline direct calls in gen.go too.
	if false {
		reflectCall(a, cgn, true)
	}
}

func ext۰reflect۰Value۰Convert(a *analysis, cgn *cgnode) {} // TODO(adonovan)

// ---------- func (Value).Elem() Value ----------

// result = v.Elem()
type rVElemConstraint struct {
	cgn    *cgnode
	v      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *rVElemConstraint) ptr() nodeid { return c.v }
func (c *rVElemConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "rVElem.result")
}
func (c *rVElemConstraint) renumber(mapping []nodeid) {
	c.v = mapping[c.v]
	c.result = mapping[c.result]
}

func (c *rVElemConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.Elem()", c.result, c.v)
}

func (c *rVElemConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		vObj := nodeid(x)
		tDyn, payload, indirect := a.taggedValue(vObj)
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		switch t := tDyn.Underlying().(type) {
		case *types.Interface:
			if a.onlineCopy(c.result, payload) {
				changed = true
			}

		case *types.Pointer:
			obj := a.makeTagged(t.Elem(), c.cgn, nil)
			a.load(obj+1, payload, 0, a.sizeof(t.Elem()))
			if a.addLabel(c.result, obj) {
				changed = true
			}
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰Value۰Elem(a *analysis, cgn *cgnode) {
	a.addConstraint(&rVElemConstraint{
		cgn:    cgn,
		v:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

func ext۰reflect۰Value۰Field(a *analysis, cgn *cgnode)           {} // TODO(adonovan)
func ext۰reflect۰Value۰FieldByIndex(a *analysis, cgn *cgnode)    {} // TODO(adonovan)
func ext۰reflect۰Value۰FieldByName(a *analysis, cgn *cgnode)     {} // TODO(adonovan)
func ext۰reflect۰Value۰FieldByNameFunc(a *analysis, cgn *cgnode) {} // TODO(adonovan)

// ---------- func (Value).Index() Value ----------

// result = v.Index()
type rVIndexConstraint struct {
	cgn    *cgnode
	v      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *rVIndexConstraint) ptr() nodeid { return c.v }
func (c *rVIndexConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "rVIndex.result")
}
func (c *rVIndexConstraint) renumber(mapping []nodeid) {
	c.v = mapping[c.v]
	c.result = mapping[c.result]
}

func (c *rVIndexConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.Index()", c.result, c.v)
}

func (c *rVIndexConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		vObj := nodeid(x)
		tDyn, payload, indirect := a.taggedValue(vObj)
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		var res nodeid
		switch t := tDyn.Underlying().(type) {
		case *types.Array:
			res = a.makeTagged(t.Elem(), c.cgn, nil)
			a.onlineCopyN(res+1, payload+1, a.sizeof(t.Elem()))

		case *types.Slice:
			res = a.makeTagged(t.Elem(), c.cgn, nil)
			a.load(res+1, payload, 1, a.sizeof(t.Elem()))

		case *types.Basic:
			if t.Kind() == types.String {
				res = a.makeTagged(types.Typ[types.Rune], c.cgn, nil)
			}
		}
		if res != 0 && a.addLabel(c.result, res) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰Value۰Index(a *analysis, cgn *cgnode) {
	a.addConstraint(&rVIndexConstraint{
		cgn:    cgn,
		v:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func (Value).Interface() Value ----------

// result = v.Interface()
type rVInterfaceConstraint struct {
	v      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *rVInterfaceConstraint) ptr() nodeid { return c.v }
func (c *rVInterfaceConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "rVInterface.result")
}
func (c *rVInterfaceConstraint) renumber(mapping []nodeid) {
	c.v = mapping[c.v]
	c.result = mapping[c.result]
}

func (c *rVInterfaceConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.Interface()", c.result, c.v)
}

func (c *rVInterfaceConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		vObj := nodeid(x)
		tDyn, payload, indirect := a.taggedValue(vObj)
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		if isInterface(tDyn) {
			if a.onlineCopy(c.result, payload) {
				a.addWork(c.result)
			}
		} else {
			if a.addLabel(c.result, vObj) {
				changed = true
			}
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰Value۰Interface(a *analysis, cgn *cgnode) {
	a.addConstraint(&rVInterfaceConstraint{
		v:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func (Value).MapIndex(Value) Value ----------

// result = v.MapIndex(_)
type rVMapIndexConstraint struct {
	cgn    *cgnode
	v      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *rVMapIndexConstraint) ptr() nodeid { return c.v }
func (c *rVMapIndexConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "rVMapIndex.result")
}
func (c *rVMapIndexConstraint) renumber(mapping []nodeid) {
	c.v = mapping[c.v]
	c.result = mapping[c.result]
}

func (c *rVMapIndexConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.MapIndex(_)", c.result, c.v)
}

func (c *rVMapIndexConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		vObj := nodeid(x)
		tDyn, m, indirect := a.taggedValue(vObj)
		tMap, _ := tDyn.Underlying().(*types.Map)
		if tMap == nil {
			continue // not a map
		}
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		obj := a.makeTagged(tMap.Elem(), c.cgn, nil)
		a.load(obj+1, m, a.sizeof(tMap.Key()), a.sizeof(tMap.Elem()))
		if a.addLabel(c.result, obj) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰Value۰MapIndex(a *analysis, cgn *cgnode) {
	a.addConstraint(&rVMapIndexConstraint{
		cgn:    cgn,
		v:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func (Value).MapKeys() []Value ----------

// result = v.MapKeys()
type rVMapKeysConstraint struct {
	cgn    *cgnode
	v      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *rVMapKeysConstraint) ptr() nodeid { return c.v }
func (c *rVMapKeysConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "rVMapKeys.result")
}
func (c *rVMapKeysConstraint) renumber(mapping []nodeid) {
	c.v = mapping[c.v]
	c.result = mapping[c.result]
}

func (c *rVMapKeysConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.MapKeys()", c.result, c.v)
}

func (c *rVMapKeysConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		vObj := nodeid(x)
		tDyn, m, indirect := a.taggedValue(vObj)
		tMap, _ := tDyn.Underlying().(*types.Map)
		if tMap == nil {
			continue // not a map
		}
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		kObj := a.makeTagged(tMap.Key(), c.cgn, nil)
		a.load(kObj+1, m, 0, a.sizeof(tMap.Key()))
		if a.addLabel(c.result, kObj) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰Value۰MapKeys(a *analysis, cgn *cgnode) {
	// Allocate an array for the result.
	obj := a.nextNode()
	T := types.NewSlice(a.reflectValueObj.Type())
	a.addNodes(sliceToArray(T), "reflect.MapKeys result")
	a.endObject(obj, cgn, nil)
	a.addressOf(T, a.funcResults(cgn.obj), obj)

	a.addConstraint(&rVMapKeysConstraint{
		cgn:    cgn,
		v:      a.funcParams(cgn.obj),
		result: obj + 1, // result is stored in array elems
	})
}

func ext۰reflect۰Value۰Method(a *analysis, cgn *cgnode)       {} // TODO(adonovan)
func ext۰reflect۰Value۰MethodByName(a *analysis, cgn *cgnode) {} // TODO(adonovan)

// ---------- func (Value).Recv(Value) Value ----------

// result, _ = v.Recv()
type rVRecvConstraint struct {
	cgn    *cgnode
	v      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *rVRecvConstraint) ptr() nodeid { return c.v }
func (c *rVRecvConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "rVRecv.result")
}
func (c *rVRecvConstraint) renumber(mapping []nodeid) {
	c.v = mapping[c.v]
	c.result = mapping[c.result]
}

func (c *rVRecvConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.Recv()", c.result, c.v)
}

func (c *rVRecvConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		vObj := nodeid(x)
		tDyn, ch, indirect := a.taggedValue(vObj)
		tChan, _ := tDyn.Underlying().(*types.Chan)
		if tChan == nil {
			continue // not a channel
		}
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		tElem := tChan.Elem()
		elemObj := a.makeTagged(tElem, c.cgn, nil)
		a.load(elemObj+1, ch, 0, a.sizeof(tElem))
		if a.addLabel(c.result, elemObj) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰Value۰Recv(a *analysis, cgn *cgnode) {
	a.addConstraint(&rVRecvConstraint{
		cgn:    cgn,
		v:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func (Value).Send(Value) ----------

// v.Send(x)
type rVSendConstraint struct {
	cgn *cgnode
	v   nodeid // (ptr)
	x   nodeid
}

func (c *rVSendConstraint) ptr() nodeid   { return c.v }
func (c *rVSendConstraint) presolve(*hvn) {}
func (c *rVSendConstraint) renumber(mapping []nodeid) {
	c.v = mapping[c.v]
	c.x = mapping[c.x]
}

func (c *rVSendConstraint) String() string {
	return fmt.Sprintf("reflect n%d.Send(n%d)", c.v, c.x)
}

func (c *rVSendConstraint) solve(a *analysis, delta *nodeset) {
	for _, x := range delta.AppendTo(a.deltaSpace) {
		vObj := nodeid(x)
		tDyn, ch, indirect := a.taggedValue(vObj)
		tChan, _ := tDyn.Underlying().(*types.Chan)
		if tChan == nil {
			continue // not a channel
		}
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		// Extract x's payload to xtmp, then store to channel.
		tElem := tChan.Elem()
		xtmp := a.addNodes(tElem, "Send.xtmp")
		a.typeAssert(tElem, xtmp, c.x, false)
		a.store(ch, xtmp, 0, a.sizeof(tElem))
	}
}

func ext۰reflect۰Value۰Send(a *analysis, cgn *cgnode) {
	params := a.funcParams(cgn.obj)
	a.addConstraint(&rVSendConstraint{
		cgn: cgn,
		v:   params,
		x:   params + 1,
	})
}

func ext۰reflect۰Value۰Set(a *analysis, cgn *cgnode) {} // TODO(adonovan)

// ---------- func (Value).SetBytes(x []byte) ----------

// v.SetBytes(x)
type rVSetBytesConstraint struct {
	cgn *cgnode
	v   nodeid // (ptr)
	x   nodeid
}

func (c *rVSetBytesConstraint) ptr() nodeid   { return c.v }
func (c *rVSetBytesConstraint) presolve(*hvn) {}
func (c *rVSetBytesConstraint) renumber(mapping []nodeid) {
	c.v = mapping[c.v]
	c.x = mapping[c.x]
}

func (c *rVSetBytesConstraint) String() string {
	return fmt.Sprintf("reflect n%d.SetBytes(n%d)", c.v, c.x)
}

func (c *rVSetBytesConstraint) solve(a *analysis, delta *nodeset) {
	for _, x := range delta.AppendTo(a.deltaSpace) {
		vObj := nodeid(x)
		tDyn, slice, indirect := a.taggedValue(vObj)
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		tSlice, ok := tDyn.Underlying().(*types.Slice)
		if ok && types.Identical(tSlice.Elem(), types.Typ[types.Uint8]) {
			if a.onlineCopy(slice, c.x) {
				a.addWork(slice)
			}
		}
	}
}

func ext۰reflect۰Value۰SetBytes(a *analysis, cgn *cgnode) {
	params := a.funcParams(cgn.obj)
	a.addConstraint(&rVSetBytesConstraint{
		cgn: cgn,
		v:   params,
		x:   params + 1,
	})
}

// ---------- func (Value).SetMapIndex(k Value, v Value) ----------

// v.SetMapIndex(key, val)
type rVSetMapIndexConstraint struct {
	cgn *cgnode
	v   nodeid // (ptr)
	key nodeid
	val nodeid
}

func (c *rVSetMapIndexConstraint) ptr() nodeid   { return c.v }
func (c *rVSetMapIndexConstraint) presolve(*hvn) {}
func (c *rVSetMapIndexConstraint) renumber(mapping []nodeid) {
	c.v = mapping[c.v]
	c.key = mapping[c.key]
	c.val = mapping[c.val]
}

func (c *rVSetMapIndexConstraint) String() string {
	return fmt.Sprintf("reflect n%d.SetMapIndex(n%d, n%d)", c.v, c.key, c.val)
}

func (c *rVSetMapIndexConstraint) solve(a *analysis, delta *nodeset) {
	for _, x := range delta.AppendTo(a.deltaSpace) {
		vObj := nodeid(x)
		tDyn, m, indirect := a.taggedValue(vObj)
		tMap, _ := tDyn.Underlying().(*types.Map)
		if tMap == nil {
			continue // not a map
		}
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		keysize := a.sizeof(tMap.Key())

		// Extract key's payload to keytmp, then store to map key.
		keytmp := a.addNodes(tMap.Key(), "SetMapIndex.keytmp")
		a.typeAssert(tMap.Key(), keytmp, c.key, false)
		a.store(m, keytmp, 0, keysize)

		// Extract val's payload to vtmp, then store to map value.
		valtmp := a.addNodes(tMap.Elem(), "SetMapIndex.valtmp")
		a.typeAssert(tMap.Elem(), valtmp, c.val, false)
		a.store(m, valtmp, keysize, a.sizeof(tMap.Elem()))
	}
}

func ext۰reflect۰Value۰SetMapIndex(a *analysis, cgn *cgnode) {
	params := a.funcParams(cgn.obj)
	a.addConstraint(&rVSetMapIndexConstraint{
		cgn: cgn,
		v:   params,
		key: params + 1,
		val: params + 2,
	})
}

func ext۰reflect۰Value۰SetPointer(a *analysis, cgn *cgnode) {} // TODO(adonovan)

// ---------- func (Value).Slice(v Value, i, j int) Value ----------

// result = v.Slice(_, _)
type rVSliceConstraint struct {
	cgn    *cgnode
	v      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *rVSliceConstraint) ptr() nodeid { return c.v }
func (c *rVSliceConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "rVSlice.result")
}
func (c *rVSliceConstraint) renumber(mapping []nodeid) {
	c.v = mapping[c.v]
	c.result = mapping[c.result]
}

func (c *rVSliceConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.Slice(_, _)", c.result, c.v)
}

func (c *rVSliceConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		vObj := nodeid(x)
		tDyn, payload, indirect := a.taggedValue(vObj)
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		var res nodeid
		switch t := tDyn.Underlying().(type) {
		case *types.Pointer:
			if tArr, ok := t.Elem().Underlying().(*types.Array); ok {
				// pointer to array
				res = a.makeTagged(types.NewSlice(tArr.Elem()), c.cgn, nil)
				if a.onlineCopy(res+1, payload) {
					a.addWork(res + 1)
				}
			}

		case *types.Array:
			// TODO(adonovan): implement addressable
			// arrays when we do indirect tagged objects.

		case *types.Slice:
			res = vObj

		case *types.Basic:
			if t == types.Typ[types.String] {
				res = vObj
			}
		}

		if res != 0 && a.addLabel(c.result, res) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰Value۰Slice(a *analysis, cgn *cgnode) {
	a.addConstraint(&rVSliceConstraint{
		cgn:    cgn,
		v:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// -------------------- Standalone reflect functions --------------------

func ext۰reflect۰Append(a *analysis, cgn *cgnode)      {} // TODO(adonovan)
func ext۰reflect۰AppendSlice(a *analysis, cgn *cgnode) {} // TODO(adonovan)
func ext۰reflect۰Copy(a *analysis, cgn *cgnode)        {} // TODO(adonovan)

// ---------- func ChanOf(ChanDir, Type) Type ----------

// result = ChanOf(dir, t)
type reflectChanOfConstraint struct {
	cgn    *cgnode
	t      nodeid // (ptr)
	result nodeid // (indirect)
	dirs   []types.ChanDir
}

func (c *reflectChanOfConstraint) ptr() nodeid { return c.t }
func (c *reflectChanOfConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "reflectChanOf.result")
}
func (c *reflectChanOfConstraint) renumber(mapping []nodeid) {
	c.t = mapping[c.t]
	c.result = mapping[c.result]
}

func (c *reflectChanOfConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.ChanOf(n%d)", c.result, c.t)
}

func (c *reflectChanOfConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		tObj := nodeid(x)
		T := a.rtypeTaggedValue(tObj)

		if typeTooHigh(T) {
			continue
		}

		for _, dir := range c.dirs {
			if a.addLabel(c.result, a.makeRtype(types.NewChan(dir, T))) {
				changed = true
			}
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

// dirMap maps reflect.ChanDir to the set of channel types generated by ChanOf.
var dirMap = [...][]types.ChanDir{
	0:               {types.SendOnly, types.RecvOnly, types.SendRecv}, // unknown
	reflect.RecvDir: {types.RecvOnly},
	reflect.SendDir: {types.SendOnly},
	reflect.BothDir: {types.SendRecv},
}

func ext۰reflect۰ChanOf(a *analysis, cgn *cgnode) {
	// If we have access to the callsite,
	// and the channel argument is a constant (as is usual),
	// only generate the requested direction.
	var dir reflect.ChanDir // unknown
	if site := cgn.callersite; site != nil {
		if c, ok := site.instr.Common().Args[0].(*ssa.Const); ok {
			v, _ := constant.Int64Val(c.Value)
			if 0 <= v && v <= int64(reflect.BothDir) {
				dir = reflect.ChanDir(v)
			}
		}
	}

	params := a.funcParams(cgn.obj)
	a.addConstraint(&reflectChanOfConstraint{
		cgn:    cgn,
		t:      params + 1,
		result: a.funcResults(cgn.obj),
		dirs:   dirMap[dir],
	})
}

// ---------- func Indirect(v Value) Value ----------

// result = Indirect(v)
type reflectIndirectConstraint struct {
	cgn    *cgnode
	v      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *reflectIndirectConstraint) ptr() nodeid { return c.v }
func (c *reflectIndirectConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "reflectIndirect.result")
}
func (c *reflectIndirectConstraint) renumber(mapping []nodeid) {
	c.v = mapping[c.v]
	c.result = mapping[c.result]
}

func (c *reflectIndirectConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.Indirect(n%d)", c.result, c.v)
}

func (c *reflectIndirectConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		vObj := nodeid(x)
		tDyn, _, _ := a.taggedValue(vObj)
		var res nodeid
		if tPtr, ok := tDyn.Underlying().(*types.Pointer); ok {
			// load the payload of the pointer's tagged object
			// into a new tagged object
			res = a.makeTagged(tPtr.Elem(), c.cgn, nil)
			a.load(res+1, vObj+1, 0, a.sizeof(tPtr.Elem()))
		} else {
			res = vObj
		}

		if a.addLabel(c.result, res) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰Indirect(a *analysis, cgn *cgnode) {
	a.addConstraint(&reflectIndirectConstraint{
		cgn:    cgn,
		v:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func MakeChan(Type) Value ----------

// result = MakeChan(typ)
type reflectMakeChanConstraint struct {
	cgn    *cgnode
	typ    nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *reflectMakeChanConstraint) ptr() nodeid { return c.typ }
func (c *reflectMakeChanConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "reflectMakeChan.result")
}
func (c *reflectMakeChanConstraint) renumber(mapping []nodeid) {
	c.typ = mapping[c.typ]
	c.result = mapping[c.result]
}

func (c *reflectMakeChanConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.MakeChan(n%d)", c.result, c.typ)
}

func (c *reflectMakeChanConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		typObj := nodeid(x)
		T := a.rtypeTaggedValue(typObj)
		tChan, ok := T.Underlying().(*types.Chan)
		if !ok || tChan.Dir() != types.SendRecv {
			continue // not a bidirectional channel type
		}

		obj := a.nextNode()
		a.addNodes(tChan.Elem(), "reflect.MakeChan.value")
		a.endObject(obj, c.cgn, nil)

		// put its address in a new T-tagged object
		id := a.makeTagged(T, c.cgn, nil)
		a.addLabel(id+1, obj)

		// flow the T-tagged object to the result
		if a.addLabel(c.result, id) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰MakeChan(a *analysis, cgn *cgnode) {
	a.addConstraint(&reflectMakeChanConstraint{
		cgn:    cgn,
		typ:    a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

func ext۰reflect۰MakeFunc(a *analysis, cgn *cgnode) {} // TODO(adonovan)

// ---------- func MakeMap(Type) Value ----------

// result = MakeMap(typ)
type reflectMakeMapConstraint struct {
	cgn    *cgnode
	typ    nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *reflectMakeMapConstraint) ptr() nodeid { return c.typ }
func (c *reflectMakeMapConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "reflectMakeMap.result")
}
func (c *reflectMakeMapConstraint) renumber(mapping []nodeid) {
	c.typ = mapping[c.typ]
	c.result = mapping[c.result]
}

func (c *reflectMakeMapConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.MakeMap(n%d)", c.result, c.typ)
}

func (c *reflectMakeMapConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		typObj := nodeid(x)
		T := a.rtypeTaggedValue(typObj)
		tMap, ok := T.Underlying().(*types.Map)
		if !ok {
			continue // not a map type
		}

		mapObj := a.nextNode()
		a.addNodes(tMap.Key(), "reflect.MakeMap.key")
		a.addNodes(tMap.Elem(), "reflect.MakeMap.value")
		a.endObject(mapObj, c.cgn, nil)

		// put its address in a new T-tagged object
		id := a.makeTagged(T, c.cgn, nil)
		a.addLabel(id+1, mapObj)

		// flow the T-tagged object to the result
		if a.addLabel(c.result, id) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰MakeMap(a *analysis, cgn *cgnode) {
	a.addConstraint(&reflectMakeMapConstraint{
		cgn:    cgn,
		typ:    a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func MakeSlice(Type) Value ----------

// result = MakeSlice(typ)
type reflectMakeSliceConstraint struct {
	cgn    *cgnode
	typ    nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *reflectMakeSliceConstraint) ptr() nodeid { return c.typ }
func (c *reflectMakeSliceConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "reflectMakeSlice.result")
}
func (c *reflectMakeSliceConstraint) renumber(mapping []nodeid) {
	c.typ = mapping[c.typ]
	c.result = mapping[c.result]
}

func (c *reflectMakeSliceConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.MakeSlice(n%d)", c.result, c.typ)
}

func (c *reflectMakeSliceConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		typObj := nodeid(x)
		T := a.rtypeTaggedValue(typObj)
		if _, ok := T.Underlying().(*types.Slice); !ok {
			continue // not a slice type
		}

		obj := a.nextNode()
		a.addNodes(sliceToArray(T), "reflect.MakeSlice")
		a.endObject(obj, c.cgn, nil)

		// put its address in a new T-tagged object
		id := a.makeTagged(T, c.cgn, nil)
		a.addLabel(id+1, obj)

		// flow the T-tagged object to the result
		if a.addLabel(c.result, id) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰MakeSlice(a *analysis, cgn *cgnode) {
	a.addConstraint(&reflectMakeSliceConstraint{
		cgn:    cgn,
		typ:    a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

func ext۰reflect۰MapOf(a *analysis, cgn *cgnode) {} // TODO(adonovan)

// ---------- func New(Type) Value ----------

// result = New(typ)
type reflectNewConstraint struct {
	cgn    *cgnode
	typ    nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *reflectNewConstraint) ptr() nodeid { return c.typ }
func (c *reflectNewConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "reflectNew.result")
}
func (c *reflectNewConstraint) renumber(mapping []nodeid) {
	c.typ = mapping[c.typ]
	c.result = mapping[c.result]
}

func (c *reflectNewConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.New(n%d)", c.result, c.typ)
}

func (c *reflectNewConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		typObj := nodeid(x)
		T := a.rtypeTaggedValue(typObj)

		// allocate new T object
		newObj := a.nextNode()
		a.addNodes(T, "reflect.New")
		a.endObject(newObj, c.cgn, nil)

		// put its address in a new *T-tagged object
		id := a.makeTagged(types.NewPointer(T), c.cgn, nil)
		a.addLabel(id+1, newObj)

		// flow the pointer to the result
		if a.addLabel(c.result, id) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰New(a *analysis, cgn *cgnode) {
	a.addConstraint(&reflectNewConstraint{
		cgn:    cgn,
		typ:    a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

func ext۰reflect۰NewAt(a *analysis, cgn *cgnode) {
	ext۰reflect۰New(a, cgn)

	// TODO(adonovan): also report dynamic calls to unsound intrinsics.
	if site := cgn.callersite; site != nil {
		a.warnf(site.pos(), "unsound: %s contains a reflect.NewAt() call", site.instr.Parent())
	}
}

// ---------- func PtrTo(Type) Type ----------

// result = PtrTo(t)
type reflectPtrToConstraint struct {
	cgn    *cgnode
	t      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *reflectPtrToConstraint) ptr() nodeid { return c.t }
func (c *reflectPtrToConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "reflectPtrTo.result")
}
func (c *reflectPtrToConstraint) renumber(mapping []nodeid) {
	c.t = mapping[c.t]
	c.result = mapping[c.result]
}

func (c *reflectPtrToConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.PtrTo(n%d)", c.result, c.t)
}

func (c *reflectPtrToConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		tObj := nodeid(x)
		T := a.rtypeTaggedValue(tObj)

		if typeTooHigh(T) {
			continue
		}

		if a.addLabel(c.result, a.makeRtype(types.NewPointer(T))) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰PtrTo(a *analysis, cgn *cgnode) {
	a.addConstraint(&reflectPtrToConstraint{
		cgn:    cgn,
		t:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

func ext۰reflect۰Select(a *analysis, cgn *cgnode) {} // TODO(adonovan)

// ---------- func SliceOf(Type) Type ----------

// result = SliceOf(t)
type reflectSliceOfConstraint struct {
	cgn    *cgnode
	t      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *reflectSliceOfConstraint) ptr() nodeid { return c.t }
func (c *reflectSliceOfConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "reflectSliceOf.result")
}
func (c *reflectSliceOfConstraint) renumber(mapping []nodeid) {
	c.t = mapping[c.t]
	c.result = mapping[c.result]
}

func (c *reflectSliceOfConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.SliceOf(n%d)", c.result, c.t)
}

func (c *reflectSliceOfConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		tObj := nodeid(x)
		T := a.rtypeTaggedValue(tObj)

		if typeTooHigh(T) {
			continue
		}

		if a.addLabel(c.result, a.makeRtype(types.NewSlice(T))) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰SliceOf(a *analysis, cgn *cgnode) {
	a.addConstraint(&reflectSliceOfConstraint{
		cgn:    cgn,
		t:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func TypeOf(v Value) Type ----------

// result = TypeOf(i)
type reflectTypeOfConstraint struct {
	cgn    *cgnode
	i      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *reflectTypeOfConstraint) ptr() nodeid { return c.i }
func (c *reflectTypeOfConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "reflectTypeOf.result")
}
func (c *reflectTypeOfConstraint) renumber(mapping []nodeid) {
	c.i = mapping[c.i]
	c.result = mapping[c.result]
}

func (c *reflectTypeOfConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.TypeOf(n%d)", c.result, c.i)
}

func (c *reflectTypeOfConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		iObj := nodeid(x)
		tDyn, _, _ := a.taggedValue(iObj)
		if a.addLabel(c.result, a.makeRtype(tDyn)) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰TypeOf(a *analysis, cgn *cgnode) {
	a.addConstraint(&reflectTypeOfConstraint{
		cgn:    cgn,
		i:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func ValueOf(interface{}) Value ----------

func ext۰reflect۰ValueOf(a *analysis, cgn *cgnode) {
	// TODO(adonovan): when we start creating indirect tagged
	// objects, we'll need to handle them specially here since
	// they must never appear in the PTS of an interface{}.
	a.copy(a.funcResults(cgn.obj), a.funcParams(cgn.obj), 1)
}

// ---------- func Zero(Type) Value ----------

// result = Zero(typ)
type reflectZeroConstraint struct {
	cgn    *cgnode
	typ    nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *reflectZeroConstraint) ptr() nodeid { return c.typ }
func (c *reflectZeroConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "reflectZero.result")
}
func (c *reflectZeroConstraint) renumber(mapping []nodeid) {
	c.typ = mapping[c.typ]
	c.result = mapping[c.result]
}

func (c *reflectZeroConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.Zero(n%d)", c.result, c.typ)
}

func (c *reflectZeroConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		typObj := nodeid(x)
		T := a.rtypeTaggedValue(typObj)

		// TODO(adonovan): if T is an interface type, we need
		// to create an indirect tagged object containing
		// new(T).  To avoid updates of such shared values,
		// we'll need another flag on indirect tagged objects
		// that marks whether they are addressable or
		// readonly, just like the reflect package does.

		// memoize using a.reflectZeros[T]
		var id nodeid
		if z := a.reflectZeros.At(T); false && z != nil {
			id = z.(nodeid)
		} else {
			id = a.makeTagged(T, c.cgn, nil)
			a.reflectZeros.Set(T, id)
		}
		if a.addLabel(c.result, id) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰Zero(a *analysis, cgn *cgnode) {
	a.addConstraint(&reflectZeroConstraint{
		cgn:    cgn,
		typ:    a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// -------------------- (*reflect.rtype) methods --------------------

// ---------- func (*rtype) Elem() Type ----------

// result = Elem(t)
type rtypeElemConstraint struct {
	cgn    *cgnode
	t      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *rtypeElemConstraint) ptr() nodeid { return c.t }
func (c *rtypeElemConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "rtypeElem.result")
}
func (c *rtypeElemConstraint) renumber(mapping []nodeid) {
	c.t = mapping[c.t]
	c.result = mapping[c.result]
}

func (c *rtypeElemConstraint) String() string {
	return fmt.Sprintf("n%d = (*reflect.rtype).Elem(n%d)", c.result, c.t)
}

func (c *rtypeElemConstraint) solve(a *analysis, delta *nodeset) {
	// Implemented by *types.{Map,Chan,Array,Slice,Pointer}.
	type hasElem interface {
		Elem() types.Type
	}
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		tObj := nodeid(x)
		T := a.nodes[tObj].obj.data.(types.Type)
		if tHasElem, ok := T.Underlying().(hasElem); ok {
			if a.addLabel(c.result, a.makeRtype(tHasElem.Elem())) {
				changed = true
			}
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰rtype۰Elem(a *analysis, cgn *cgnode) {
	a.addConstraint(&rtypeElemConstraint{
		cgn:    cgn,
		t:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func (*rtype) Field(int) StructField ----------
// ---------- func (*rtype) FieldByName(string) (StructField, bool) ----------

// result = FieldByName(t, name)
// result = Field(t, _)
type rtypeFieldByNameConstraint struct {
	cgn    *cgnode
	name   string // name of field; "" for unknown
	t      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *rtypeFieldByNameConstraint) ptr() nodeid { return c.t }
func (c *rtypeFieldByNameConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result+3), "rtypeFieldByName.result.Type")
}
func (c *rtypeFieldByNameConstraint) renumber(mapping []nodeid) {
	c.t = mapping[c.t]
	c.result = mapping[c.result]
}

func (c *rtypeFieldByNameConstraint) String() string {
	return fmt.Sprintf("n%d = (*reflect.rtype).FieldByName(n%d, %q)", c.result, c.t, c.name)
}

func (c *rtypeFieldByNameConstraint) solve(a *analysis, delta *nodeset) {
	// type StructField struct {
	// 0	__identity__
	// 1	Name      string
	// 2	PkgPath   string
	// 3	Type      Type
	// 4	Tag       StructTag
	// 5	Offset    uintptr
	// 6	Index     []int
	// 7	Anonymous bool
	// }

	for _, x := range delta.AppendTo(a.deltaSpace) {
		tObj := nodeid(x)
		T := a.nodes[tObj].obj.data.(types.Type)
		tStruct, ok := T.Underlying().(*types.Struct)
		if !ok {
			continue // not a struct type
		}

		n := tStruct.NumFields()
		for i := 0; i < n; i++ {
			f := tStruct.Field(i)
			if c.name == "" || c.name == f.Name() {

				// a.offsetOf(Type) is 3.
				if id := c.result + 3; a.addLabel(id, a.makeRtype(f.Type())) {
					a.addWork(id)
				}
				// TODO(adonovan): StructField.Index should be non-nil.
			}
		}
	}
}

func ext۰reflect۰rtype۰FieldByName(a *analysis, cgn *cgnode) {
	// If we have access to the callsite,
	// and the argument is a string constant,
	// return only that field.
	var name string
	if site := cgn.callersite; site != nil {
		if c, ok := site.instr.Common().Args[0].(*ssa.Const); ok {
			name = constant.StringVal(c.Value)
		}
	}

	a.addConstraint(&rtypeFieldByNameConstraint{
		cgn:    cgn,
		name:   name,
		t:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

func ext۰reflect۰rtype۰Field(a *analysis, cgn *cgnode) {
	// No-one ever calls Field with a constant argument,
	// so we don't specialize that case.
	a.addConstraint(&rtypeFieldByNameConstraint{
		cgn:    cgn,
		t:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

func ext۰reflect۰rtype۰FieldByIndex(a *analysis, cgn *cgnode)    {} // TODO(adonovan)
func ext۰reflect۰rtype۰FieldByNameFunc(a *analysis, cgn *cgnode) {} // TODO(adonovan)

// ---------- func (*rtype) In/Out(i int) Type ----------

// result = In/Out(t, i)
type rtypeInOutConstraint struct {
	cgn    *cgnode
	t      nodeid // (ptr)
	result nodeid // (indirect)
	out    bool
	i      int // -ve if not a constant
}

func (c *rtypeInOutConstraint) ptr() nodeid { return c.t }
func (c *rtypeInOutConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "rtypeInOut.result")
}
func (c *rtypeInOutConstraint) renumber(mapping []nodeid) {
	c.t = mapping[c.t]
	c.result = mapping[c.result]
}

func (c *rtypeInOutConstraint) String() string {
	return fmt.Sprintf("n%d = (*reflect.rtype).InOut(n%d, %d)", c.result, c.t, c.i)
}

func (c *rtypeInOutConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		tObj := nodeid(x)
		T := a.nodes[tObj].obj.data.(types.Type)
		sig, ok := T.Underlying().(*types.Signature)
		if !ok {
			continue // not a func type
		}

		tuple := sig.Params()
		if c.out {
			tuple = sig.Results()
		}
		for i, n := 0, tuple.Len(); i < n; i++ {
			if c.i < 0 || c.i == i {
				if a.addLabel(c.result, a.makeRtype(tuple.At(i).Type())) {
					changed = true
				}
			}
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰rtype۰InOut(a *analysis, cgn *cgnode, out bool) {
	// If we have access to the callsite,
	// and the argument is an int constant,
	// return only that parameter.
	index := -1
	if site := cgn.callersite; site != nil {
		if c, ok := site.instr.Common().Args[0].(*ssa.Const); ok {
			v, _ := constant.Int64Val(c.Value)
			index = int(v)
		}
	}
	a.addConstraint(&rtypeInOutConstraint{
		cgn:    cgn,
		t:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
		out:    out,
		i:      index,
	})
}

func ext۰reflect۰rtype۰In(a *analysis, cgn *cgnode) {
	ext۰reflect۰rtype۰InOut(a, cgn, false)
}

func ext۰reflect۰rtype۰Out(a *analysis, cgn *cgnode) {
	ext۰reflect۰rtype۰InOut(a, cgn, true)
}

// ---------- func (*rtype) Key() Type ----------

// result = Key(t)
type rtypeKeyConstraint struct {
	cgn    *cgnode
	t      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *rtypeKeyConstraint) ptr() nodeid { return c.t }
func (c *rtypeKeyConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result), "rtypeKey.result")
}
func (c *rtypeKeyConstraint) renumber(mapping []nodeid) {
	c.t = mapping[c.t]
	c.result = mapping[c.result]
}

func (c *rtypeKeyConstraint) String() string {
	return fmt.Sprintf("n%d = (*reflect.rtype).Key(n%d)", c.result, c.t)
}

func (c *rtypeKeyConstraint) solve(a *analysis, delta *nodeset) {
	changed := false
	for _, x := range delta.AppendTo(a.deltaSpace) {
		tObj := nodeid(x)
		T := a.nodes[tObj].obj.data.(types.Type)
		if tMap, ok := T.Underlying().(*types.Map); ok {
			if a.addLabel(c.result, a.makeRtype(tMap.Key())) {
				changed = true
			}
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰rtype۰Key(a *analysis, cgn *cgnode) {
	a.addConstraint(&rtypeKeyConstraint{
		cgn:    cgn,
		t:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func (*rtype) Method(int) (Method, bool) ----------
// ---------- func (*rtype) MethodByName(string) (Method, bool) ----------

// result = MethodByName(t, name)
// result = Method(t, _)
type rtypeMethodByNameConstraint struct {
	cgn    *cgnode
	name   string // name of method; "" for unknown
	t      nodeid // (ptr)
	result nodeid // (indirect)
}

func (c *rtypeMethodByNameConstraint) ptr() nodeid { return c.t }
func (c *rtypeMethodByNameConstraint) presolve(h *hvn) {
	h.markIndirect(onodeid(c.result+3), "rtypeMethodByName.result.Type")
	h.markIndirect(onodeid(c.result+4), "rtypeMethodByName.result.Func")
}
func (c *rtypeMethodByNameConstraint) renumber(mapping []nodeid) {
	c.t = mapping[c.t]
	c.result = mapping[c.result]
}

func (c *rtypeMethodByNameConstraint) String() string {
	return fmt.Sprintf("n%d = (*reflect.rtype).MethodByName(n%d, %q)", c.result, c.t, c.name)
}

// changeRecv returns sig with Recv prepended to Params().
func changeRecv(sig *types.Signature) *types.Signature {
	params := sig.Params()
	n := params.Len()
	p2 := make([]*types.Var, n+1)
	p2[0] = sig.Recv()
	for i := 0; i < n; i++ {
		p2[i+1] = params.At(i)
	}
	return types.NewSignature(nil, types.NewTuple(p2...), sig.Results(), sig.Variadic())
}

func (c *rtypeMethodByNameConstraint) solve(a *analysis, delta *nodeset) {
	for _, x := range delta.AppendTo(a.deltaSpace) {
		tObj := nodeid(x)
		T := a.nodes[tObj].obj.data.(types.Type)

		isIface := isInterface(T)

		// We don't use Lookup(c.name) when c.name != "" to avoid
		// ambiguity: >1 unexported methods could match.
		mset := a.prog.MethodSets.MethodSet(T)
		for i, n := 0, mset.Len(); i < n; i++ {
			sel := mset.At(i)
			if c.name == "" || c.name == sel.Obj().Name() {
				// type Method struct {
				// 0     __identity__
				// 1	Name    string
				// 2	PkgPath string
				// 3	Type    Type
				// 4	Func    Value
				// 5	Index   int
				// }

				var sig *types.Signature
				var fn *ssa.Function
				if isIface {
					sig = sel.Type().(*types.Signature)
				} else {
					fn = a.prog.MethodValue(sel)
					// move receiver to params[0]
					sig = changeRecv(fn.Signature)
				}

				// a.offsetOf(Type) is 3.
				if id := c.result + 3; a.addLabel(id, a.makeRtype(sig)) {
					a.addWork(id)
				}
				if fn != nil {
					// a.offsetOf(Func) is 4.
					if id := c.result + 4; a.addLabel(id, a.objectNode(nil, fn)) {
						a.addWork(id)
					}
				}
			}
		}
	}
}

func ext۰reflect۰rtype۰MethodByName(a *analysis, cgn *cgnode) {
	// If we have access to the callsite,
	// and the argument is a string constant,
	// return only that method.
	var name string
	if site := cgn.callersite; site != nil {
		if c, ok := site.instr.Common().Args[0].(*ssa.Const); ok {
			name = constant.StringVal(c.Value)
		}
	}

	a.addConstraint(&rtypeMethodByNameConstraint{
		cgn:    cgn,
		name:   name,
		t:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

func ext۰reflect۰rtype۰Method(a *analysis, cgn *cgnode) {
	// No-one ever calls Method with a constant argument,
	// so we don't specialize that case.
	a.addConstraint(&rtypeMethodByNameConstraint{
		cgn:    cgn,
		t:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// typeHeight returns the "height" of the type, which is roughly
// speaking the number of chan, map, pointer and slice type constructors
// at the root of T; these are the four type kinds that can be created
// via reflection.  Chan and map constructors are counted as double the
// height of slice and pointer constructors since they are less often
// deeply nested.
//
// The solver rules for type constructors must somehow bound the set of
// types they create to ensure termination of the algorithm in cases
// where the output of a type constructor flows to its input, e.g.
//
//	func f(t reflect.Type) {
//		f(reflect.PtrTo(t))
//	}
//
// It does this by limiting the type height to k, but this still leaves
// a potentially exponential (4^k) number of of types that may be
// enumerated in pathological cases.
func typeHeight(T types.Type) int {
	switch T := T.(type) {
	case *types.Chan:
		return 2 + typeHeight(T.Elem())
	case *types.Map:
		k := typeHeight(T.Key())
		v := typeHeight(T.Elem())
		if v > k {
			k = v // max(k, v)
		}
		return 2 + k
	case *types.Slice:
		return 1 + typeHeight(T.Elem())
	case *types.Pointer:
		return 1 + typeHeight(T.Elem())
	}
	return 0
}

func typeTooHigh(T types.Type) bool {
	return typeHeight(T) > 3
}
