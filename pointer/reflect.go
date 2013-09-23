package pointer

// This file implements the generation and resolution rules for
// constraints arising from the use of reflection in the target
// program.  See doc.go for explanation of the representation.
//
// For consistency, the names of all parameters match those of the
// actual functions in the "reflect" package.
//
// TODO(adonovan): fix: most of the reflect API permits implicit
// conversions due to assignability, e.g. m.MapIndex(k) is ok if T(k)
// is assignable to T(M).key.  It's not yet clear how best to model
// that; perhaps a more lenient version of typeAssertConstraint is
// needed.
//
// To avoid proliferation of equivalent labels, instrinsics should
// memoize as much as possible, like TypeOf and Zero do for their
// tagged objects.
//
// TODO(adonovan): all {} functions are TODO.

import (
	"fmt"
	"go/ast"

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

// result = v.Interface()
type rVInterfaceConstraint struct {
	v      nodeid // (ptr)
	result nodeid
}

func (c *rVInterfaceConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.Interface()", c.result, c.v)
}

func (c *rVInterfaceConstraint) ptr() nodeid {
	return c.v
}

func (c *rVInterfaceConstraint) solve(a *analysis, _ *node, delta nodeset) {
	resultPts := &a.nodes[c.result].pts
	changed := false
	for vObj := range delta {
		tDyn, _, indirect := a.taggedValue(vObj)
		if tDyn == nil {
			panic("not a tagged object")
		}
		if indirect {
			// TODO(adonovan): we'll need to implement this
			// when we start creating indirect tagged objects.
			panic("indirect tagged object")
		}

		if resultPts.add(vObj) {
			changed = true
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
	result nodeid
}

func (c *rVMapIndexConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.MapIndex(_)", c.result, c.v)
}

func (c *rVMapIndexConstraint) ptr() nodeid {
	return c.v
}

func (c *rVMapIndexConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for vObj := range delta {
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
		a.loadOffset(obj+1, m, a.sizeof(tMap.Key()), a.sizeof(tMap.Elem()))
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
	result nodeid
}

func (c *rVMapKeysConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.MapKeys()", c.result, c.v)
}

func (c *rVMapKeysConstraint) ptr() nodeid {
	return c.v
}

func (c *rVMapKeysConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for vObj := range delta {
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
		a.load(kObj+1, m, a.sizeof(tMap.Key()))
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
	a.addNodes(types.NewArray(a.reflectValueObj.Type(), 1), "reflect.MapKeys result")
	a.endObject(obj, cgn, nil)
	a.addressOf(a.funcResults(cgn.obj), obj)

	a.addConstraint(&rVMapKeysConstraint{
		cgn:    cgn,
		v:      a.funcParams(cgn.obj),
		result: obj + 1, // result is stored in array elems
	})
}

func ext۰reflect۰Value۰Method(a *analysis, cgn *cgnode)       {}
func ext۰reflect۰Value۰MethodByName(a *analysis, cgn *cgnode) {}

// ---------- func (Value).Recv(Value) ----------

// result, _ = v.Recv()
type rVRecvConstraint struct {
	cgn    *cgnode
	v      nodeid // (ptr)
	result nodeid
}

func (c *rVRecvConstraint) String() string {
	return fmt.Sprintf("n%d = reflect n%d.Recv()", c.result, c.v)
}

func (c *rVRecvConstraint) ptr() nodeid {
	return c.v
}

func (c *rVRecvConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for vObj := range delta {
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
		a.load(elemObj+1, ch, a.sizeof(tElem))
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

func (c *rVSendConstraint) String() string {
	return fmt.Sprintf("reflect n%d.Send(n%d)", c.v, c.x)
}

func (c *rVSendConstraint) ptr() nodeid {
	return c.v
}

func (c *rVSendConstraint) solve(a *analysis, _ *node, delta nodeset) {
	for vObj := range delta {
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
		a.typeAssert(tElem, xtmp, c.x)
		a.store(ch, xtmp, a.sizeof(tElem))
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

func ext۰reflect۰Value۰Set(a *analysis, cgn *cgnode)      {}
func ext۰reflect۰Value۰SetBytes(a *analysis, cgn *cgnode) {}

// ---------- func (Value).SetMapIndex(k Value, v Value) ----------

// v.SetMapIndex(key, val)
type rVSetMapIndexConstraint struct {
	cgn *cgnode
	v   nodeid // (ptr)
	key nodeid
	val nodeid
}

func (c *rVSetMapIndexConstraint) String() string {
	return fmt.Sprintf("reflect n%d.SetMapIndex(n%d, n%d)", c.v, c.key, c.val)
}

func (c *rVSetMapIndexConstraint) ptr() nodeid {
	return c.v
}

func (c *rVSetMapIndexConstraint) solve(a *analysis, _ *node, delta nodeset) {
	for vObj := range delta {
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
		a.typeAssert(tMap.Key(), keytmp, c.key)
		a.store(m, keytmp, keysize)

		// Extract val's payload to vtmp, then store to map value.
		valtmp := a.addNodes(tMap.Elem(), "SetMapIndex.valtmp")
		a.typeAssert(tMap.Elem(), valtmp, c.val)
		a.storeOffset(m, valtmp, keysize, a.sizeof(tMap.Elem()))
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

func ext۰reflect۰Value۰SetPointer(a *analysis, cgn *cgnode) {}
func ext۰reflect۰Value۰Slice(a *analysis, cgn *cgnode)      {}

// -------------------- Standalone reflect functions --------------------

func ext۰reflect۰Append(a *analysis, cgn *cgnode)      {}
func ext۰reflect۰AppendSlice(a *analysis, cgn *cgnode) {}
func ext۰reflect۰Copy(a *analysis, cgn *cgnode)        {}

// ---------- func ChanOf(ChanDir, Type) Type ----------

// result = ChanOf(_, t)
type reflectChanOfConstraint struct {
	cgn    *cgnode
	t      nodeid // (ptr)
	result nodeid
}

func (c *reflectChanOfConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.ChanOf(n%d)", c.result, c.t)
}

func (c *reflectChanOfConstraint) ptr() nodeid {
	return c.t
}

func (c *reflectChanOfConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for tObj := range delta {
		T := a.rtypeTaggedValue(tObj)
		// TODO(adonovan): use only the channel direction
		// provided at the callsite, if constant.
		for _, dir := range []ast.ChanDir{1, 2, 3} {
			if a.addLabel(c.result, a.makeRtype(types.NewChan(dir, T))) {
				changed = true
			}
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰ChanOf(a *analysis, cgn *cgnode) {
	params := a.funcParams(cgn.obj)
	a.addConstraint(&reflectChanOfConstraint{
		cgn:    cgn,
		t:      params + 1,
		result: a.funcResults(cgn.obj),
	})
}

// ---------- func Indirect(v Value) Value ----------

// result = Indirect(v)
type reflectIndirectConstraint struct {
	cgn    *cgnode
	v      nodeid // (ptr)
	result nodeid
}

func (c *reflectIndirectConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.Indirect(n%d)", c.result, c.v)
}

func (c *reflectIndirectConstraint) ptr() nodeid {
	return c.v
}

func (c *reflectIndirectConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for vObj := range delta {
		tDyn, _, _ := a.taggedValue(vObj)
		if tDyn == nil {
			panic("not a tagged value")
		}

		var res nodeid
		if tPtr, ok := tDyn.Underlying().(*types.Pointer); ok {
			// load the payload of the pointer's tagged object
			// into a new tagged object
			res = a.makeTagged(tPtr.Elem(), c.cgn, nil)
			a.load(res+1, vObj+1, a.sizeof(tPtr.Elem()))
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
	result nodeid
}

func (c *reflectMakeChanConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.MakeChan(n%d)", c.result, c.typ)
}

func (c *reflectMakeChanConstraint) ptr() nodeid {
	return c.typ
}

func (c *reflectMakeChanConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for typObj := range delta {
		T := a.rtypeTaggedValue(typObj)
		tChan, ok := T.Underlying().(*types.Chan)
		if !ok || tChan.Dir() != ast.SEND|ast.RECV {
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

func ext۰reflect۰MakeFunc(a *analysis, cgn *cgnode) {}

// ---------- func MakeMap(Type) Value ----------

// result = MakeMap(typ)
type reflectMakeMapConstraint struct {
	cgn    *cgnode
	typ    nodeid // (ptr)
	result nodeid
}

func (c *reflectMakeMapConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.MakeMap(n%d)", c.result, c.typ)
}

func (c *reflectMakeMapConstraint) ptr() nodeid {
	return c.typ
}

func (c *reflectMakeMapConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for typObj := range delta {
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

func ext۰reflect۰MakeSlice(a *analysis, cgn *cgnode) {}
func ext۰reflect۰MapOf(a *analysis, cgn *cgnode)     {}

// ---------- func New(Type) Value ----------

// result = New(typ)
type reflectNewConstraint struct {
	cgn    *cgnode
	typ    nodeid // (ptr)
	result nodeid
}

func (c *reflectNewConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.New(n%d)", c.result, c.typ)
}

func (c *reflectNewConstraint) ptr() nodeid {
	return c.typ
}

func (c *reflectNewConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for typObj := range delta {
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

	// TODO(adonovan): make it easier to report errors of this form,
	// which includes the callsite:
	// a.warnf("unsound: main.reflectNewAt contains a reflect.NewAt() call")
	a.warnf(cgn.Func().Pos(), "unsound: reflect.NewAt() call")
}

func ext۰reflect۰PtrTo(a *analysis, cgn *cgnode)   {}
func ext۰reflect۰Select(a *analysis, cgn *cgnode)  {}
func ext۰reflect۰SliceOf(a *analysis, cgn *cgnode) {}

// ---------- func TypeOf(v Value) Type ----------

// result = TypeOf(i)
type reflectTypeOfConstraint struct {
	cgn    *cgnode
	i      nodeid // (ptr)
	result nodeid
}

func (c *reflectTypeOfConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.TypeOf(n%d)", c.result, c.i)
}

func (c *reflectTypeOfConstraint) ptr() nodeid {
	return c.i
}

func (c *reflectTypeOfConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for iObj := range delta {
		tDyn, _, _ := a.taggedValue(iObj)
		if tDyn == nil {
			panic("not a tagged value")
		}

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
	result nodeid
}

func (c *reflectZeroConstraint) String() string {
	return fmt.Sprintf("n%d = reflect.Zero(n%d)", c.result, c.typ)
}

func (c *reflectZeroConstraint) ptr() nodeid {
	return c.typ
}

func (c *reflectZeroConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for typObj := range delta {
		T := a.rtypeTaggedValue(typObj)

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
	result nodeid
}

func (c *rtypeElemConstraint) String() string {
	return fmt.Sprintf("n%d = (*reflect.rtype).Elem(n%d)", c.result, c.t)
}

func (c *rtypeElemConstraint) ptr() nodeid {
	return c.t
}

func (c *rtypeElemConstraint) solve(a *analysis, _ *node, delta nodeset) {
	// Implemented by *types.{Map,Chan,Array,Slice,Pointer}.
	type hasElem interface {
		Elem() types.Type
	}
	changed := false
	for tObj := range delta {
		T := a.nodes[tObj].obj.rtype
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

func ext۰reflect۰rtype۰Field(a *analysis, cgn *cgnode)           {}
func ext۰reflect۰rtype۰FieldByIndex(a *analysis, cgn *cgnode)    {}
func ext۰reflect۰rtype۰FieldByName(a *analysis, cgn *cgnode)     {}
func ext۰reflect۰rtype۰FieldByNameFunc(a *analysis, cgn *cgnode) {}

// ---------- func (*rtype) In/Out() Type ----------

// result = In/Out(t)
type rtypeInOutConstraint struct {
	cgn    *cgnode
	t      nodeid // (ptr)
	result nodeid
	out    bool
}

func (c *rtypeInOutConstraint) String() string {
	return fmt.Sprintf("n%d = (*reflect.rtype).InOut(n%d)", c.result, c.t)
}

func (c *rtypeInOutConstraint) ptr() nodeid {
	return c.t
}

func (c *rtypeInOutConstraint) solve(a *analysis, _ *node, delta nodeset) {
	changed := false
	for tObj := range delta {
		T := a.nodes[tObj].obj.rtype
		sig, ok := T.Underlying().(*types.Signature)
		if !ok {
			continue // not a func type
		}

		tuple := sig.Params()
		if c.out {
			tuple = sig.Results()
		}
		// TODO(adonovan): when a function is analyzed
		// context-sensitively, we should be able to see its
		// caller's actual parameter's ssa.Values.  Refactor
		// the intrinsic mechanism to allow this.  Then if the
		// value is an int const K, skip the loop and use
		// tuple.At(K).
		for i, n := 0, tuple.Len(); i < n; i++ {
			if a.addLabel(c.result, a.makeRtype(tuple.At(i).Type())) {
				changed = true
			}
		}
	}
	if changed {
		a.addWork(c.result)
	}
}

func ext۰reflect۰rtype۰InOut(a *analysis, cgn *cgnode, out bool) {
	a.addConstraint(&rtypeInOutConstraint{
		cgn:    cgn,
		t:      a.funcParams(cgn.obj),
		result: a.funcResults(cgn.obj),
		out:    out,
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
	for tObj := range delta {
		T := a.nodes[tObj].obj.rtype
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

func ext۰reflect۰rtype۰Method(a *analysis, cgn *cgnode)       {}
func ext۰reflect۰rtype۰MethodByName(a *analysis, cgn *cgnode) {}
