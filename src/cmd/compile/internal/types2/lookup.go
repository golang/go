// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements various field and method lookup functions.

package types2

import (
	"bytes"
	"strings"
)

// LookupSelection selects the field or method whose ID is Id(pkg,
// name), on a value of type T. If addressable is set, T is the type
// of an addressable variable (this matters only for method lookups).
// T must not be nil.
//
// If the selection is valid:
//
//   - [Selection.Obj] returns the field ([Var]) or method ([Func]);
//   - [Selection.Indirect] reports whether there were any pointer
//     indirections on the path to the field or method.
//   - [Selection.Index] returns the index sequence, defined below.
//
// The last index entry is the field or method index in the (possibly
// embedded) type where the entry was found, either:
//
//  1. the list of declared methods of a named type; or
//  2. the list of all methods (method set) of an interface type; or
//  3. the list of fields of a struct type.
//
// The earlier index entries are the indices of the embedded struct
// fields traversed to get to the found entry, starting at depth 0.
//
// See also [LookupFieldOrMethod], which returns the components separately.
func LookupSelection(T Type, addressable bool, pkg *Package, name string) (Selection, bool) {
	obj, index, indirect := LookupFieldOrMethod(T, addressable, pkg, name)
	var kind SelectionKind
	switch obj.(type) {
	case nil:
		return Selection{}, false
	case *Func:
		kind = MethodVal
	case *Var:
		kind = FieldVal
	default:
		panic(obj) // can't happen
	}
	return Selection{kind, T, obj, index, indirect}, true
}

// Internal use of LookupFieldOrMethod: If the obj result is a method
// associated with a concrete (non-interface) type, the method's signature
// may not be fully set up. Call Checker.objDecl(obj, nil) before accessing
// the method's type.

// LookupFieldOrMethod looks up a field or method with given package and name
// in T and returns the corresponding *Var or *Func, an index sequence, and a
// bool indicating if there were any pointer indirections on the path to the
// field or method. If addressable is set, T is the type of an addressable
// variable (only matters for method lookups). T must not be nil.
//
// The last index entry is the field or method index in the (possibly embedded)
// type where the entry was found, either:
//
//  1. the list of declared methods of a named type; or
//  2. the list of all methods (method set) of an interface type; or
//  3. the list of fields of a struct type.
//
// The earlier index entries are the indices of the embedded struct fields
// traversed to get to the found entry, starting at depth 0.
//
// If no entry is found, a nil object is returned. In this case, the returned
// index and indirect values have the following meaning:
//
//   - If index != nil, the index sequence points to an ambiguous entry
//     (the same name appeared more than once at the same embedding level).
//
//   - If indirect is set, a method with a pointer receiver type was found
//     but there was no pointer on the path from the actual receiver type to
//     the method's formal receiver base type, nor was the receiver addressable.
//
// See also [LookupSelection], which returns the result as a [Selection].
func LookupFieldOrMethod(T Type, addressable bool, pkg *Package, name string) (obj Object, index []int, indirect bool) {
	if T == nil {
		panic("LookupFieldOrMethod on nil type")
	}
	return lookupFieldOrMethod(T, addressable, pkg, name, false)
}

// lookupFieldOrMethod is like LookupFieldOrMethod but with the additional foldCase parameter
// (see Object.sameId for the meaning of foldCase).
func lookupFieldOrMethod(T Type, addressable bool, pkg *Package, name string, foldCase bool) (obj Object, index []int, indirect bool) {
	// Methods cannot be associated to a named pointer type.
	// (spec: "The type denoted by T is called the receiver base type;
	// it must not be a pointer or interface type and it must be declared
	// in the same package as the method.").
	// Thus, if we have a named pointer type, proceed with the underlying
	// pointer type but discard the result if it is a method since we would
	// not have found it for T (see also go.dev/issue/8590).
	if t := asNamed(T); t != nil {
		if p, _ := t.Underlying().(*Pointer); p != nil {
			obj, index, indirect = lookupFieldOrMethodImpl(p, false, pkg, name, foldCase)
			if _, ok := obj.(*Func); ok {
				return nil, nil, false
			}
			return
		}
	}

	obj, index, indirect = lookupFieldOrMethodImpl(T, addressable, pkg, name, foldCase)

	// If we didn't find anything and if we have a type parameter with a common underlying
	// type, see if there is a matching field (but not a method, those need to be declared
	// explicitly in the constraint). If the constraint is a named pointer type (see above),
	// we are ok here because only fields are accepted as results.
	const enableTParamFieldLookup = false // see go.dev/issue/51576
	if enableTParamFieldLookup && obj == nil && isTypeParam(T) {
		if t, _ := commonUnder(T, nil); t != nil {
			obj, index, indirect = lookupFieldOrMethodImpl(t, addressable, pkg, name, foldCase)
			if _, ok := obj.(*Var); !ok {
				obj, index, indirect = nil, nil, false // accept fields (variables) only
			}
		}
	}
	return
}

// lookupFieldOrMethodImpl is the implementation of lookupFieldOrMethod.
// Notably, in contrast to lookupFieldOrMethod, it won't find struct fields
// in base types of defined (*Named) pointer types T. For instance, given
// the declaration:
//
//	type T *struct{f int}
//
// lookupFieldOrMethodImpl won't find the field f in the defined (*Named) type T
// (methods on T are not permitted in the first place).
//
// Thus, lookupFieldOrMethodImpl should only be called by lookupFieldOrMethod
// and missingMethod (the latter doesn't care about struct fields).
//
// The resulting object may not be fully type-checked.
func lookupFieldOrMethodImpl(T Type, addressable bool, pkg *Package, name string, foldCase bool) (obj Object, index []int, indirect bool) {
	// WARNING: The code in this function is extremely subtle - do not modify casually!

	if name == "_" {
		return // blank fields/methods are never found
	}

	// Importantly, we must not call under before the call to deref below (nor
	// does deref call under), as doing so could incorrectly result in finding
	// methods of the pointer base type when T is a (*Named) pointer type.
	typ, isPtr := deref(T)

	// *typ where typ is an interface (incl. a type parameter) has no methods.
	if isPtr {
		if _, ok := under(typ).(*Interface); ok {
			return
		}
	}

	// Go spec: basic types don't have methods. But this toolchain supports "magic"
	// operator methods (e.g. _add/_sub/...) via a front-end rewrite pass.
	//
	// To allow generic constraints such as `interface{ _add(T) T }` to be satisfied
	// by numeric basic types (int/float64/etc.), we synthesize these magic methods
	// on-demand during lookup and constraint checking.
	//
	// Note: This is intentionally limited to a small, well-known set of magic
	// method names and only for numeric basic types, to avoid accidentally making
	// basic types satisfy arbitrary method constraints.
	if !isPtr {
		// Helper: parse synthesized _init overload names.
		// Supported forms:
		//   _init            -> 0 args
		//   _init_int        -> 1 arg (int-ish)
		//   _init_int_int64  -> 2 args (int-ish, int-ish)
		// Where "int-ish" is any integer basic type token used by the overload rewriter.
		parseInitArgs := func(mname string) (argTypes []*Basic, ok bool) {
			// For compatibility with constraints like `interface{ _init(int) }`,
			// treat plain "_init" as the 1-arg "int" form for builtin containers.
			// (Note: we can't represent both _init() and _init(int) under the same
			// name without an overload suffix, so we pick the common 1-arg form.)
			if mname == "_init" {
				return []*Basic{Typ[Int]}, true
			}
			const prefix = "_init_"
			if !strings.HasPrefix(mname, prefix) {
				return nil, false
			}
			rest := mname[len(prefix):]
			if rest == "" {
				return nil, false
			}
			parts := strings.Split(rest, "_")
			argTypes = make([]*Basic, 0, len(parts))
			for _, p := range parts {
				var bt *Basic
				switch p {
				case "int":
					bt = Typ[Int]
				case "int8":
					bt = Typ[Int8]
				case "int16":
					bt = Typ[Int16]
				case "int32":
					bt = Typ[Int32]
				case "int64":
					bt = Typ[Int64]
				case "uint":
					bt = Typ[Uint]
				case "uint8":
					bt = Typ[Uint8]
				case "uint16":
					bt = Typ[Uint16]
				case "uint32":
					bt = Typ[Uint32]
				case "uint64":
					bt = Typ[Uint64]
				case "uintptr":
					bt = Typ[Uintptr]
				case "byte":
					bt = Typ[Byte]
				case "rune":
					bt = Typ[Rune]
				default:
					return nil, false
				}
				argTypes = append(argTypes, bt)
			}
			return argTypes, true
		}

		if b, _ := under(typ).(*Basic); b != nil {
			// 定义三种类型的操作符集合

			// 1. 一元运算符 (Unary): func() T
			//    - _pos (+a), _neg (-a), _invert (^a)
			isUnary := false
			switch name {
			case "_pos", "_neg", "_invert":
				isUnary = true
			}

			// 2. 比较运算符 (Compare): func(T) bool
			//    - _eq, _ne, _lt, _le, _gt, _ge
			isCompare := false
			switch name {
			case "_eq", "_ne", "_lt", "_le", "_gt", "_ge":
				isCompare = true
			}

			// 3. 二元算术/位运算符 (Binary): func(T) T
			//    - _add, _sub, _mul, _div, _mod
			//    - _and, _or, _xor, _bitclear
			//    - _lshift, _rshift
			//    - 以及它们的反向版本 _radd...
			isBinary := false
			switch name {
			case "_add", "_sub", "_mul", "_div", "_mod",
				"_and", "_or", "_xor", "_bitclear",
				"_lshift", "_rshift",
				"_radd", "_rsub", "_rmul", "_rdiv", "_rmod",
				"_rand", "_ror", "_rxor", "_rbitclear",
				"_rlshift", "_rrshift":
				isBinary = true
			}

			// 检查类型是否支持这些操作
			info := b.Info()
			isNumeric := info&IsNumeric != 0
			isString := info&IsString != 0
			isBoolean := info&IsBoolean != 0

			// 只有数值或字符串才支持
			if isNumeric || isString || isBoolean {
				isValidMagic := true

				if isString {
					if name != "_add" && !isCompare {
						isValidMagic = false
					}
				}

				if isBoolean {
					// bool
					if name != "_eq" && name != "_ne" {
						isValidMagic = false
					}
				}

				if isValidMagic && (isUnary || isCompare || isBinary) {
					// 1. 构造接收者 (Receiver)
					recv := NewParam(nopos, pkg, "", typ)
					recv.SetKind(RecvVar)

					// 2. 构造参数列表 (Params)
					var params []*Var
					if isUnary {
						// 一元运算没有参数: func (T) _neg() ...
						params = nil
					} else {
						// 二元/比较运算有一个参数: func (T) _add(other T) ...
						arg := NewParam(nopos, pkg, "", typ)
						params = []*Var{arg}
					}

					// 3. 构造返回值列表 (Results)
					var results []*Var
					if isCompare {
						// 比较运算返回 bool: ... bool
						// 注意：这里使用 Typ[Bool] 获取 types2 内部的 bool 类型
						boolType := Typ[Bool]
						res := NewParam(nopos, pkg, "", boolType)
						res.SetKind(ResultVar)
						results = []*Var{res}
					} else {
						// 其他运算返回自身类型 T: ... T
						res := NewParam(nopos, pkg, "", typ)
						res.SetKind(ResultVar)
						results = []*Var{res}
					}

					// 4. 合成函数签名
					sig := NewSignatureType(recv, nil, nil, NewTuple(params...), NewTuple(results...), false)

					// 5. 返回对象
					return NewFunc(nopos, pkg, name, sig), []int{0}, false
				}
			}

		}

		// =========================================================
		// B. 新增: Slice 支持 _getitem / _setitem
		//    func (s []E) _getitem(index int) E
		//    func (s []E) _setitem(index int, val E)
		// =========================================================
		if s, ok := under(typ).(*Slice); ok {
			if name == "_getitem" || name == "_setitem" {
				// 1. 构造接收者 (Receiver): []E
				recv := NewParam(nopos, pkg, "", typ)
				recv.SetKind(RecvVar)

				// 2. 准备类型
				intType := Typ[Int]  // 索引类型
				elemType := s.Elem() // 元素类型

				var params []*Var
				var results []*Var

				if name == "_getitem" {
					// signature: func(int) Elem
					arg1 := NewParam(nopos, pkg, "index", intType)
					params = []*Var{arg1}

					res := NewParam(nopos, pkg, "", elemType)
					res.SetKind(ResultVar)
					results = []*Var{res}
				} else { // _setitem
					// signature: func(int, Elem)
					arg1 := NewParam(nopos, pkg, "index", intType)
					arg2 := NewParam(nopos, pkg, "val", elemType)
					params = []*Var{arg1, arg2}

					// 无返回值
					results = nil
				}

				// 3. 合成函数签名
				sig := NewSignatureType(recv, nil, nil, NewTuple(params...), NewTuple(results...), false)

				// 4. 返回对象
				return NewFunc(nopos, pkg, name, sig), []int{0}, false
			}
		}

		// =========================================================
		// C. 新增: Map 支持 _getitem / _setitem
		//    func (m map[K]V) _getitem(key K) V
		//    func (m map[K]V) _setitem(key K, val V)
		// =========================================================
		if m, ok := under(typ).(*Map); ok {
			if name == "_getitem" || name == "_setitem" {
				// 1. 构造接收者 (Receiver): map[K]V
				recv := NewParam(nopos, pkg, "", typ)
				recv.SetKind(RecvVar)

				// 2. 准备类型
				keyType := m.Key()
				elemType := m.Elem()

				var params []*Var
				var results []*Var

				if name == "_getitem" {
					// signature: func(Key) Elem
					// 注意：这里为了简化泛型约束匹配，_getitem 只返回 Value
					// 忽略了 comma-ok 模式
					arg1 := NewParam(nopos, pkg, "key", keyType)
					params = []*Var{arg1}

					res := NewParam(nopos, pkg, "", elemType)
					res.SetKind(ResultVar)
					results = []*Var{res}
				} else { // _setitem
					// signature: func(Key, Elem)
					arg1 := NewParam(nopos, pkg, "key", keyType)
					arg2 := NewParam(nopos, pkg, "val", elemType)
					params = []*Var{arg1, arg2}

					// 无返回值
					results = nil
				}

				// 3. 合成函数签名
				sig := NewSignatureType(recv, nil, nil, NewTuple(params...), NewTuple(results...), false)

				// 4. 返回对象
				return NewFunc(nopos, pkg, name, sig), []int{0}, false
			}
		}

		// =========================================================
		// D. 新增: Slice/Map/Chan 合成 _init（用于让它们满足构造器约束）
		//
		// - Slice: _init(int) / _init(int,int)  (对应 make([]T, len) / make([]T, len, cap))
		// - Map:   _init() / _init(int)        (对应 make(map[K]V) / make(map[K]V, hint))
		// - Chan:  _init() / _init(int)        (对应 make(chan T) / make(chan T, buf))
		//
		// 注意：这些是“合成方法”，只用于 lookup/约束检查，让类型参数约束可以写成
		// interface{ _init(int) } 之类；并不意味着用户能在运行时真正调用到一个 Go 方法体。
		// =========================================================
		if argBasics, ok := parseInitArgs(name); ok {
			// 1. 构造接收者：用当前 typ（保留 named 类型形态，和上面的 _getitem/_setitem 一致）
			recv := NewParam(nopos, pkg, "", typ)
			recv.SetKind(RecvVar)

			makeSig := func() *Signature {
				var params []*Var
				for _, bt := range argBasics {
					p := NewParam(nopos, pkg, "", bt)
					params = append(params, p)
				}
				// 无返回值：让它直接匹配 interface{ _init(...) } 的常见写法
				return NewSignatureType(recv, nil, nil, NewTuple(params...), nil, false)
			}

			switch under(typ).(type) {
			case *Slice:
				if len(argBasics) == 1 || len(argBasics) == 2 {
					return NewFunc(nopos, pkg, name, makeSig()), []int{0}, false
				}
			case *Map:
				// map 允许 0 或 1 个整数参数
				if len(argBasics) == 0 || len(argBasics) == 1 {
					return NewFunc(nopos, pkg, name, makeSig()), []int{0}, false
				}
			case *Chan:
				// chan 允许 0 或 1 个整数参数
				if len(argBasics) == 0 || len(argBasics) == 1 {
					return NewFunc(nopos, pkg, name, makeSig()), []int{0}, false
				}
			}
		}
	}

	// Start with typ as single entry at shallowest depth.
	current := []embeddedType{{typ, nil, isPtr, false}}

	// seen tracks named types that we have seen already, allocated lazily.
	// Used to avoid endless searches in case of recursive types.
	//
	// We must use a lookup on identity rather than a simple map[*Named]bool as
	// instantiated types may be identical but not equal.
	var seen instanceLookup

	// search current depth
	for len(current) > 0 {
		var next []embeddedType // embedded types found at current depth

		// look for (pkg, name) in all types at current depth
		for _, e := range current {
			typ := e.typ

			// If we have a named type, we may have associated methods.
			// Look for those first.
			if named := asNamed(typ); named != nil {
				if alt := seen.lookup(named); alt != nil {
					// We have seen this type before, at a more shallow depth
					// (note that multiples of this type at the current depth
					// were consolidated before). The type at that depth shadows
					// this same type at the current depth, so we can ignore
					// this one.
					continue
				}
				seen.add(named)

				// look for a matching attached method
				if i, m := named.lookupMethod(pkg, name, foldCase); m != nil {
					// potential match
					// caution: method may not have a proper signature yet
					index = concat(e.index, i)
					if obj != nil || e.multiples {
						return nil, index, false // collision
					}
					obj = m
					indirect = e.indirect
					continue // we can't have a matching field or interface method
				}
			}

			switch t := under(typ).(type) {
			case *Struct:
				// look for a matching field and collect embedded types
				for i, f := range t.fields {
					if f.sameId(pkg, name, foldCase) {
						assert(f.typ != nil)
						index = concat(e.index, i)
						if obj != nil || e.multiples {
							return nil, index, false // collision
						}
						obj = f
						indirect = e.indirect
						continue // we can't have a matching interface method
					}
					// Collect embedded struct fields for searching the next
					// lower depth, but only if we have not seen a match yet
					// (if we have a match it is either the desired field or
					// we have a name collision on the same depth; in either
					// case we don't need to look further).
					// Embedded fields are always of the form T or *T where
					// T is a type name. If e.typ appeared multiple times at
					// this depth, f.typ appears multiple times at the next
					// depth.
					if obj == nil && f.embedded {
						typ, isPtr := deref(f.typ)
						// TODO(gri) optimization: ignore types that can't
						// have fields or methods (only Named, Struct, and
						// Interface types need to be considered).
						next = append(next, embeddedType{typ, concat(e.index, i), e.indirect || isPtr, e.multiples})
					}
				}

			case *Interface:
				// look for a matching method (interface may be a type parameter)
				if i, m := t.typeSet().LookupMethod(pkg, name, foldCase); m != nil {
					assert(m.typ != nil)
					index = concat(e.index, i)
					if obj != nil || e.multiples {
						return nil, index, false // collision
					}
					obj = m
					indirect = e.indirect
				}
			}
		}

		if obj != nil {
			// found a potential match
			// spec: "A method call x.m() is valid if the method set of (the type of) x
			//        contains m and the argument list can be assigned to the parameter
			//        list of m. If x is addressable and &x's method set contains m, x.m()
			//        is shorthand for (&x).m()".
			if f, _ := obj.(*Func); f != nil {
				// determine if method has a pointer receiver
				if f.hasPtrRecv() && !indirect && !addressable {
					return nil, nil, true // pointer/addressable receiver required
				}
			}
			return
		}

		current = consolidateMultiples(next)
	}

	return nil, nil, false // not found
}

// embeddedType represents an embedded type
type embeddedType struct {
	typ       Type
	index     []int // embedded field indices, starting with index at depth 0
	indirect  bool  // if set, there was a pointer indirection on the path to this field
	multiples bool  // if set, typ appears multiple times at this depth
}

// consolidateMultiples collects multiple list entries with the same type
// into a single entry marked as containing multiples. The result is the
// consolidated list.
func consolidateMultiples(list []embeddedType) []embeddedType {
	if len(list) <= 1 {
		return list // at most one entry - nothing to do
	}

	n := 0                     // number of entries w/ unique type
	prev := make(map[Type]int) // index at which type was previously seen
	for _, e := range list {
		if i, found := lookupType(prev, e.typ); found {
			list[i].multiples = true
			// ignore this entry
		} else {
			prev[e.typ] = n
			list[n] = e
			n++
		}
	}
	return list[:n]
}

func lookupType(m map[Type]int, typ Type) (int, bool) {
	// fast path: maybe the types are equal
	if i, found := m[typ]; found {
		return i, true
	}

	for t, i := range m {
		if Identical(t, typ) {
			return i, true
		}
	}

	return 0, false
}

type instanceLookup struct {
	// buf is used to avoid allocating the map m in the common case of a small
	// number of instances.
	buf [3]*Named
	m   map[*Named][]*Named
}

func (l *instanceLookup) lookup(inst *Named) *Named {
	for _, t := range l.buf {
		if t != nil && Identical(inst, t) {
			return t
		}
	}
	for _, t := range l.m[inst.Origin()] {
		if Identical(inst, t) {
			return t
		}
	}
	return nil
}

func (l *instanceLookup) add(inst *Named) {
	for i, t := range l.buf {
		if t == nil {
			l.buf[i] = inst
			return
		}
	}
	if l.m == nil {
		l.m = make(map[*Named][]*Named)
	}
	insts := l.m[inst.Origin()]
	l.m[inst.Origin()] = append(insts, inst)
}

// MissingMethod returns (nil, false) if V implements T, otherwise it
// returns a missing method required by T and whether it is missing or
// just has the wrong type: either a pointer receiver or wrong signature.
//
// For non-interface types V, or if static is set, V implements T if all
// methods of T are present in V. Otherwise (V is an interface and static
// is not set), MissingMethod only checks that methods of T which are also
// present in V have matching types (e.g., for a type assertion x.(T) where
// x is of interface type V).
func MissingMethod(V Type, T *Interface, static bool) (method *Func, wrongType bool) {
	return (*Checker)(nil).missingMethod(V, T, static, Identical, nil)
}

// missingMethod is like MissingMethod but accepts a *Checker as receiver,
// a comparator equivalent for type comparison, and a *string for error causes.
// The receiver may be nil if missingMethod is invoked through an exported
// API call (such as MissingMethod), i.e., when all methods have been type-
// checked.
// The underlying type of T must be an interface; T (rather than its under-
// lying type) is used for better error messages (reported through *cause).
// The comparator is used to compare signatures.
// If a method is missing and cause is not nil, *cause describes the error.
func (check *Checker) missingMethod(V, T Type, static bool, equivalent func(x, y Type) bool, cause *string) (method *Func, wrongType bool) {
	methods := under(T).(*Interface).typeSet().methods // T must be an interface
	if len(methods) == 0 {
		return nil, false
	}

	const (
		ok = iota
		notFound
		wrongName
		unexported
		wrongSig
		ambigSel
		ptrRecv
		field
	)

	state := ok
	var m *Func // method on T we're trying to implement
	var f *Func // method on V, if found (state is one of ok, wrongName, wrongSig)

	if u, _ := under(V).(*Interface); u != nil {
		tset := u.typeSet()
		for _, m = range methods {
			_, f = tset.LookupMethod(m.pkg, m.name, false)

			if f == nil {
				if !static {
					continue
				}
				state = notFound
				break
			}

			if !equivalentOrInit(equivalent, f, m) {
				state = wrongSig
				break
			}
		}
	} else {
		for _, m = range methods {
			obj, index, indirect := lookupFieldOrMethodImpl(V, false, m.pkg, m.name, false)

			// check if m is ambiguous, on *V, or on V with case-folding
			if obj == nil {
				switch {
				case index != nil:
					state = ambigSel
				case indirect:
					// Compiler extension: treat _init as "constructor-like".
					// For the purpose of constraint satisfaction, allow a pointer-receiver
					// _init method on *V to satisfy an _init requirement on V.
					//
					// This is used together with the "constructor make(T, ...)" extension
					// which always allocates a *T anyway.
					if m != nil && m.name == "_init" {
						if ok := initMethodOnPtrSatisfies(V, m, equivalent); ok {
							// treat as satisfied; continue checking other methods
							continue
						}
					}
					state = ptrRecv
				default:
					state = notFound
					obj, _, _ = lookupFieldOrMethodImpl(V, false, m.pkg, m.name, true /* fold case */)
					f, _ = obj.(*Func)
					if f != nil {
						state = wrongName
						if f.name == m.name {
							// If the names are equal, f must be unexported
							// (otherwise the package wouldn't matter).
							state = unexported
						}
					}
				}
				break
			}

			// we must have a method (not a struct field)
			f, _ = obj.(*Func)
			if f == nil {
				state = field
				break
			}

			// methods may not have a fully set up signature yet
			if check != nil {
				check.objDecl(f, nil)
			}

			if !equivalentOrInit(equivalent, f, m) {
				state = wrongSig
				break
			}
		}
	}

	if state == ok {
		return nil, false
	}

	if cause != nil {
		if f != nil {
			// This method may be formatted in funcString below, so must have a fully
			// set up signature.
			if check != nil {
				check.objDecl(f, nil)
			}
		}
		switch state {
		case notFound:
			switch {
			case isInterfacePtr(V):
				*cause = "(" + check.interfacePtrError(V) + ")"
			case isInterfacePtr(T):
				*cause = "(" + check.interfacePtrError(T) + ")"
			default:
				*cause = check.sprintf("(missing method %s)", m.Name())
			}
		case wrongName:
			fs, ms := check.funcString(f, false), check.funcString(m, false)
			*cause = check.sprintf("(missing method %s)\n\t\thave %s\n\t\twant %s", m.Name(), fs, ms)
		case unexported:
			*cause = check.sprintf("(unexported method %s)", m.Name())
		case wrongSig:
			fs, ms := check.funcString(f, false), check.funcString(m, false)
			if fs == ms {
				// Don't report "want Foo, have Foo".
				// Add package information to disambiguate (go.dev/issue/54258).
				fs, ms = check.funcString(f, true), check.funcString(m, true)
			}
			if fs == ms {
				// We still have "want Foo, have Foo".
				// This is most likely due to different type parameters with
				// the same name appearing in the instantiated signatures
				// (go.dev/issue/61685).
				// Rather than reporting this misleading error cause, for now
				// just point out that the method signature is incorrect.
				// TODO(gri) should find a good way to report the root cause
				*cause = check.sprintf("(wrong type for method %s)", m.Name())
				break
			}
			*cause = check.sprintf("(wrong type for method %s)\n\t\thave %s\n\t\twant %s", m.Name(), fs, ms)
		case ambigSel:
			*cause = check.sprintf("(ambiguous selector %s.%s)", V, m.Name())
		case ptrRecv:
			*cause = check.sprintf("(method %s has pointer receiver)", m.Name())
		case field:
			*cause = check.sprintf("(%s.%s is a field, not a method)", V, m.Name())
		default:
			panic("unreachable")
		}
	}

	return m, state == wrongSig || state == ptrRecv
}

// equivalentOrInit wraps the usual signature comparator with a narrow compatibility
// rule for "_init": allow an implementation method to have a single return value
// equal to its receiver type, even if the interface method has no results.
func equivalentOrInit(equivalent func(x, y Type) bool, impl, want *Func) bool {
	if equivalent(impl.typ, want.typ) {
		return true
	}
	if want == nil || want.name != "_init" || impl == nil {
		return false
	}
	return initSigCompatible(impl.typ, want.typ)
}

func initSigCompatible(impl, want Type) bool {
	isig, _ := impl.(*Signature)
	wsig, _ := want.(*Signature)
	if isig == nil || wsig == nil {
		return false
	}
	// Compare parameters (receiver excluded).
	if isig.Params().Len() != wsig.Params().Len() {
		return false
	}
	for i := 0; i < isig.Params().Len(); i++ {
		if !Identical(isig.Params().At(i).typ, wsig.Params().At(i).typ) {
			return false
		}
	}
	// Want must have no results.
	if wsig.Results().Len() != 0 {
		return false
	}
	// Impl may have 0 results, or 1 result equal to its receiver type.
	switch isig.Results().Len() {
	case 0:
		return true
	case 1:
		recv := isig.Recv()
		if recv == nil {
			return false
		}
		return Identical(isig.Results().At(0).typ, recv.typ)
	default:
		return false
	}
}

func initMethodOnPtrSatisfies(V Type, want *Func, equivalent func(x, y Type) bool) bool {
	if want == nil {
		return false
	}
	// Look up _init on *V and verify it matches the wanted signature under initSigCompatible.
	pv := &Pointer{base: V}
	obj, _, _ := lookupFieldOrMethodImpl(pv, false, want.pkg, want.name, false)
	impl, _ := obj.(*Func)
	if impl == nil {
		return false
	}
	if equivalent(impl.typ, want.typ) {
		return true
	}
	return initSigCompatible(impl.typ, want.typ)
}

// hasAllMethods is similar to checkMissingMethod but instead reports whether all methods are present.
// If V is not a valid type, or if it is a struct containing embedded fields with invalid types, the
// result is true because it is not possible to say with certainty whether a method is missing or not
// (an embedded field may have the method in question).
// If the result is false and cause is not nil, *cause describes the error.
// Use hasAllMethods to avoid follow-on errors due to incorrect types.
func (check *Checker) hasAllMethods(V, T Type, static bool, equivalent func(x, y Type) bool, cause *string) bool {
	if !isValid(V) {
		return true // we don't know anything about V, assume it implements T
	}
	m, _ := check.missingMethod(V, T, static, equivalent, cause)
	return m == nil || hasInvalidEmbeddedFields(V, nil)
}

// hasInvalidEmbeddedFields reports whether T is a struct (or a pointer to a struct) that contains
// (directly or indirectly) embedded fields with invalid types.
func hasInvalidEmbeddedFields(T Type, seen map[*Struct]bool) bool {
	if S, _ := under(derefStructPtr(T)).(*Struct); S != nil && !seen[S] {
		if seen == nil {
			seen = make(map[*Struct]bool)
		}
		seen[S] = true
		for _, f := range S.fields {
			if f.embedded && (!isValid(f.typ) || hasInvalidEmbeddedFields(f.typ, seen)) {
				return true
			}
		}
	}
	return false
}

func isInterfacePtr(T Type) bool {
	p, _ := under(T).(*Pointer)
	return p != nil && IsInterface(p.base)
}

// check may be nil.
func (check *Checker) interfacePtrError(T Type) string {
	assert(isInterfacePtr(T))
	if p, _ := under(T).(*Pointer); isTypeParam(p.base) {
		return check.sprintf("type %s is pointer to type parameter, not type parameter", T)
	}
	return check.sprintf("type %s is pointer to interface, not interface", T)
}

// funcString returns a string of the form name + signature for f.
// check may be nil.
func (check *Checker) funcString(f *Func, pkgInfo bool) string {
	buf := bytes.NewBufferString(f.name)
	var qf Qualifier
	if check != nil && !pkgInfo {
		qf = check.qualifier
	}
	w := newTypeWriter(buf, qf)
	w.pkgInfo = pkgInfo
	w.paramNames = false
	w.signature(f.typ.(*Signature))
	return buf.String()
}

// assertableTo reports whether a value of type V can be asserted to have type T.
// The receiver may be nil if assertableTo is invoked through an exported API call
// (such as AssertableTo), i.e., when all methods have been type-checked.
// The underlying type of V must be an interface.
// If the result is false and cause is not nil, *cause describes the error.
// TODO(gri) replace calls to this function with calls to newAssertableTo.
func (check *Checker) assertableTo(V, T Type, cause *string) bool {
	// no static check is required if T is an interface
	// spec: "If T is an interface type, x.(T) asserts that the
	//        dynamic type of x implements the interface T."
	if IsInterface(T) {
		return true
	}
	// TODO(gri) fix this for generalized interfaces
	return check.hasAllMethods(T, V, false, Identical, cause)
}

// newAssertableTo reports whether a value of type V can be asserted to have type T.
// It also implements behavior for interfaces that currently are only permitted
// in constraint position (we have not yet defined that behavior in the spec).
// The underlying type of V must be an interface.
// If the result is false and cause is not nil, *cause is set to the error cause.
func (check *Checker) newAssertableTo(V, T Type, cause *string) bool {
	// no static check is required if T is an interface
	// spec: "If T is an interface type, x.(T) asserts that the
	//        dynamic type of x implements the interface T."
	if IsInterface(T) {
		return true
	}
	return check.implements(T, V, false, cause)
}

// deref dereferences typ if it is a *Pointer (but not a *Named type
// with an underlying pointer type!) and returns its base and true.
// Otherwise it returns (typ, false).
func deref(typ Type) (Type, bool) {
	if p, _ := Unalias(typ).(*Pointer); p != nil {
		// p.base should never be nil, but be conservative
		if p.base == nil {
			if debug {
				panic("pointer with nil base type (possibly due to an invalid cyclic declaration)")
			}
			return Typ[Invalid], true
		}
		return p.base, true
	}
	return typ, false
}

// derefStructPtr dereferences typ if it is a (named or unnamed) pointer to a
// (named or unnamed) struct and returns its base. Otherwise it returns typ.
func derefStructPtr(typ Type) Type {
	if p, _ := under(typ).(*Pointer); p != nil {
		if _, ok := under(p.base).(*Struct); ok {
			return p.base
		}
	}
	return typ
}

// concat returns the result of concatenating list and i.
// The result does not share its underlying array with list.
func concat(list []int, i int) []int {
	var t []int
	t = append(t, list...)
	return append(t, i)
}

// fieldIndex returns the index for the field with matching package and name, or a value < 0.
// See Object.sameId for the meaning of foldCase.
func fieldIndex(fields []*Var, pkg *Package, name string, foldCase bool) int {
	if name != "_" {
		for i, f := range fields {
			if f.sameId(pkg, name, foldCase) {
				return i
			}
		}
	}
	return -1
}

// methodIndex returns the index of and method with matching package and name, or (-1, nil).
// See Object.sameId for the meaning of foldCase.
func methodIndex(methods []*Func, pkg *Package, name string, foldCase bool) (int, *Func) {
	if name != "_" {
		for i, m := range methods {
			if m.sameId(pkg, name, foldCase) {
				return i, m
			}
		}
	}
	return -1, nil
}
