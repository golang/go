package pointer

// This file implements the generation and resolution rules for
// constraints arising from the use of reflection in the target
// program.  See doc.go for explanation of the representation.
//
// TODO(adonovan): fix: most of the reflect API permits implicit
// conversions due to assignability, e.g. m.MapIndex(k) is ok if T(k)
// is assignable to T(M).key.  It's not yet clear how best to model
// that.
//
// To avoid proliferation of equivalent labels, instrinsics should
// memoize as much as possible, like TypeOf and Zero do for their
// tagged objects.
//
// TODO(adonovan): all {} functions are TODO.

import (
	"fmt"

	"code.google.com/p/go.tools/go/types"
)

// -------------------- (reflect.Value) --------------------

func ext۰reflect۰Value۰Addr(a *analysis, cgn *cgnode)            {}
func ext۰reflect۰Value۰Bytes(a *analysis, cgn *cgnode)           {}
func ext۰reflect۰Value۰Call(a *analysis, cgn *cgnode)            {}
func ext۰reflect۰Value۰CallSlice(a *analysis, cgn *cgnode)       {}
func ext۰reflect۰Value۰Convert(a *analysis, cgn *cgnode)         {}
func ext۰reflect۰Value۰Elem(a *analysis, cgn *cgnode)            {}
func ext۰reflect۰Value۰Field(a *analysis, cgn *cgnode)           {}
func ext۰reflect۰Value۰FieldByIndex(a *analysis, cgn *cgnode)    {}
func ext۰reflect۰Value۰FieldByName(a *analysis, cgn *cgnode)     {}
func ext۰reflect۰Value۰FieldByNameFunc(a *analysis, cgn *cgnode) {}
func ext۰reflect۰Value۰Index(a *analysis, cgn *cgnode)           {}

// ---------- func (Value).Interface() Value ----------

// result = rv.Interface()
type rVInterfaceConstraint struct {
	rv     nodeid // (ptr)
	result nodeid
}

func (c *rVInterfaceConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.Interface()", c.result, c.rv)
}

func (c *rVInterfaceConstraint) ptr() nodeid {
	return c.rv
}

func (c *rVInterfaceConstraint) solve(a *analysis, _ *node, delta nodeset) {
	resultPts := &a.nodes[c.result].pts
	changed := false
	for obj := range delta {
		tDyn, _, indirect := a.taggedValue(obj)
		if tDyn == nil {
			panic("not a tagged object")
		}
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		if resultPts.add(obj) {
			changed = true
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰Value۰Interface(a *analysis, cgn *cgnode) {
	a.addConstraint(&rVInterfaceConstraint{
		rv:     a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func (Value).MapIndex(Value) Value ----------

// result = rv.MapIndex(key)
type rVMapIndexConstraint struct {
	cgn    *cgnode
	rv     nodeid // (ptr)
	result nodeid
}

func (c *rVMapIndexConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.MapIndex(_)", c.result, c.rv)
}

func (c *rVMapIndexConstraint) ptr() nodeid {
	return c.rv
}

func (c *rVMapIndexConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for obj := range delta {
		tDyn, m, indirect := a.taggedValue(obj)
		tMap, _ := tDyn.(*types.Map)
		if tMap == nil {
			continue // not a map
		}
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		vObj := a.makeTagged(tMap.Elem(), c.cgn, nil)
		a.loadOffset(vObj+1, m, a.sizeof(tMap.Key()), a.sizeof(tMap.Elem()))
		if a.nodes[c.result].pts.add(vObj) {
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
		rv:     a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func (Value).MapKeys() []Value ----------

// result = rv.MapKeys()
type rVMapKeysConstraint struct {
	cgn    *cgnode
	rv     nodeid // (ptr)
	result nodeid
}

func (c *rVMapKeysConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.MapKeys()", c.result, c.rv)
}

func (c *rVMapKeysConstraint) ptr() nodeid {
	return c.rv
}

func (c *rVMapKeysConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for obj := range delta {
		tDyn, m, indirect := a.taggedValue(obj)
		tMap, _ := tDyn.(*types.Map)
		if tMap == nil {
			continue // not a map
		}
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		kObj := a.makeTagged(tMap.Key(), c.cgn, nil)
		a.load(kObj+1, m, a.sizeof(tMap.Key()))
		if a.nodes[c.result].pts.add(kObj) {
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
	a.addNodes(types.NewArray(a.reflectValueObj.Type(), 1), "reflect.MapKeys result")
	a.endObject(obj, cgn, nil)
	a.addressOf(a.funcResults(cgn.obj), obj)

	// resolution rule attached to rv
	a.addConstraint(&rVMapKeysConstraint{
		cgn:    cgn,
		rv:     a.funcParams(cgn.obj),
		result: obj + 1, // result is stored in array elems
	})
}

func ext۰reflect۰Value۰Method(a *analysis, cgn *cgnode)       {}
func ext۰reflect۰Value۰MethodByName(a *analysis, cgn *cgnode) {}
func ext۰reflect۰Value۰Set(a *analysis, cgn *cgnode)          {}
func ext۰reflect۰Value۰SetBytes(a *analysis, cgn *cgnode)     {}

// ---------- func (Value).SetMapIndex(k Value, v Value) ----------

// rv.SetMapIndex(k, v)
type rVSetMapIndexConstraint struct {
	cgn *cgnode
	rv  nodeid // (ptr)
	k   nodeid
	v   nodeid
}

func (c *rVSetMapIndexConstraint) String() string {
	return fmt.Sprintf("reflect n%d.SetMapIndex(n%d, n%d)", c.rv, c.k, c.v)
}

func (c *rVSetMapIndexConstraint) ptr() nodeid {
	return c.rv
}

func (c *rVSetMapIndexConstraint) solve(a *analysis, _ *node, delta nodeset) {
	for obj := range delta {
		tDyn, m, indirect := a.taggedValue(obj)
		tMap, _ := tDyn.(*types.Map)
		if tMap == nil {
			continue // not a map
		}
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		ksize := a.sizeof(tMap.Key())

		// Extract k Value's payload to ktmp, then store to map key.
		ktmp := a.addNodes(tMap.Key(), "SetMapIndex.ktmp")
		a.addConstraint(&typeAssertConstraint{tMap.Key(), ktmp, c.k})
		a.store(m, ktmp, ksize)

		// Extract v Value's payload to vtmp, then store to map value.
		vtmp := a.addNodes(tMap.Elem(), "SetMapIndex.vtmp")
		a.addConstraint(&typeAssertConstraint{tMap.Elem(), vtmp, c.v})
		a.storeOffset(m, vtmp, ksize, a.sizeof(tMap.Elem()))
	}
}

func ext۰reflect۰Value۰SetMapIndex(a *analysis, cgn *cgnode) {
	// resolution rule attached to rv
	rv := a.funcParams(cgn.obj)
	a.addConstraint(&rVSetMapIndexConstraint{
		cgn: cgn,
		rv:  rv,
		k:   rv + 1,
		v:   rv + 2,
	})
}

func ext۰reflect۰Value۰SetPointer(a *analysis, cgn *cgnode) {}
func ext۰reflect۰Value۰Slice(a *analysis, cgn *cgnode)      {}

// -------------------- Standalone reflect functions --------------------

func ext۰reflect۰Append(a *analysis, cgn *cgnode)      {}
func ext۰reflect۰AppendSlice(a *analysis, cgn *cgnode) {}
func ext۰reflect۰Copy(a *analysis, cgn *cgnode)        {}
func ext۰reflect۰ChanOf(a *analysis, cgn *cgnode)      {}
func ext۰reflect۰Indirect(a *analysis, cgn *cgnode)    {}
func ext۰reflect۰MakeChan(a *analysis, cgn *cgnode)    {}
func ext۰reflect۰MakeFunc(a *analysis, cgn *cgnode)    {}
func ext۰reflect۰MakeMap(a *analysis, cgn *cgnode)     {}
func ext۰reflect۰MakeSlice(a *analysis, cgn *cgnode)   {}
func ext۰reflect۰MapOf(a *analysis, cgn *cgnode)       {}
func ext۰reflect۰New(a *analysis, cgn *cgnode)         {}
func ext۰reflect۰NewAt(a *analysis, cgn *cgnode)       {}
func ext۰reflect۰PtrTo(a *analysis, cgn *cgnode)       {}
func ext۰reflect۰Select(a *analysis, cgn *cgnode)      {}
func ext۰reflect۰SliceOf(a *analysis, cgn *cgnode)     {}

// ---------- func TypeOf(v Value) Type ----------

// result = TypeOf(v)
type reflectTypeOfConstraint struct {
	cgn    *cgnode
	v      nodeid // (ptr)
	result nodeid
}

func (c *reflectTypeOfConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.TypeOf(n%d)", c.result, c.v)
}

func (c *reflectTypeOfConstraint) ptr() nodeid {
	return c.v
}

func (c *reflectTypeOfConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for obj := range delta {
		tDyn, _, _ := a.taggedValue(obj)
		if tDyn == nil {
			panic("not a tagged value")
		}

		if a.nodes[c.result].pts.add(a.makeRtype(tDyn)) {
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
		v:      a.funcParams(cgn.obj),
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

// result = Zero(t)
type reflectZeroConstraint struct {
	cgn    *cgnode
	t      nodeid // (ptr)
	result nodeid
}

func (c *reflectZeroConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.Zero(n%d)", c.result, c.t)
}

func (c *reflectZeroConstraint) ptr() nodeid {
	return c.t
}

func (c *reflectZeroConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for obj := range delta {
		tDyn, v, _ := a.taggedValue(obj)
		if tDyn != a.reflectRtype {
			panic("not a *reflect.rtype-tagged value")
		}
		T := a.nodes[v].typ

		// memoize using a.reflectZeros[T]
		var id nodeid
		if z := a.reflectZeros.At(T); false && z != nil {
			id = z.(nodeid)
		} else {
			id = a.makeTagged(T, c.cgn, nil)
			a.reflectZeros.Set(T, id)
		}
		if a.nodes[c.result].pts.add(id) {
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
		t:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
	})
}

// -------------------- (*reflect.rtype) methods --------------------

// ---------- func (*rtype) Elem() Type ----------

// result = Elem(t)
type rtypeElemConstraint struct {
	cgn    *cgnode
	t      nodeid // (ptr)
	result nodeid
}

func (c *rtypeElemConstraint) String() string {
	return fmt.Sprintf("n%d = (*reflect.rtype).Elem(n%d)", c.result, c.t)
}

func (c *rtypeElemConstraint) ptr() nodeid {
	return c.t
}

func (c *rtypeElemConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for obj := range delta {
		T := a.nodes[obj].typ // assume obj is an *rtype

		// Works for *types.{Map,Chan,Array,Slice,Pointer}.
		if T, ok := T.Underlying().(interface {
			Elem() types.Type
		}); ok {
			if a.nodes[c.result].pts.add(a.makeRtype(T.Elem())) {
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

func ext۰reflect۰rtype۰Field(a *analysis, cgn *cgnode)           {}
func ext۰reflect۰rtype۰FieldByIndex(a *analysis, cgn *cgnode)    {}
func ext۰reflect۰rtype۰FieldByName(a *analysis, cgn *cgnode)     {}
func ext۰reflect۰rtype۰FieldByNameFunc(a *analysis, cgn *cgnode) {}
func ext۰reflect۰rtype۰In(a *analysis, cgn *cgnode)              {}

// ---------- func (*rtype) Key() Type ----------

// result = Key(t)
type rtypeKeyConstraint struct {
	cgn    *cgnode
	t      nodeid // (ptr)
	result nodeid
}

func (c *rtypeKeyConstraint) String() string {
	return fmt.Sprintf("n%d = (*reflect.rtype).Key(n%d)", c.result, c.t)
}

func (c *rtypeKeyConstraint) ptr() nodeid {
	return c.t
}

func (c *rtypeKeyConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for obj := range delta {
		T := a.nodes[obj].typ // assume obj is an *rtype

		if tMap, ok := T.Underlying().(*types.Map); ok {
			if a.nodes[c.result].pts.add(a.makeRtype(tMap.Key())) {
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

func ext۰reflect۰rtype۰Method(a *analysis, cgn *cgnode)       {}
func ext۰reflect۰rtype۰MethodByName(a *analysis, cgn *cgnode) {}
func ext۰reflect۰rtype۰Out(a *analysis, cgn *cgnode)          {}
