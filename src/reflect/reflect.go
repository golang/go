package reflect

// Method returns the i'th method in the type's method set.
// It panics if i is out of range.
func (t Type) Method(i int) Method {
	if t.Kind() == Interface {
		tt := (*interfaceType)(unsafe.Pointer(t.common()))
		if uint(i) >= uint(len(tt.Methods)) {
			panic("reflect: Method index out of range")
		}
		m := &tt.Methods[i]
		return Method{
			Name:    tt.nameOff(m.Name).Name(),
			PkgPath: tt.nameOff(m.Name).PkgPath(),
			Type:    toType(tt.typeOff(m.Typ)),
			Func:    toType(tt.typeOff(m.Tfn)),
			Index:   i,
		}
	}

	tt := t.common()
	ms := tt.ExportedMethods()
	if uint(i) >= uint(len(ms)) {
		panic("reflect: Method index out of range")
	}
	m := ms[i]
	return Method{
		Name:    nameOffFor(tt, m.Name).Name(),
		PkgPath: nameOffFor(tt, m.Name).PkgPath(),
		Type:    toType(typeOffFor(tt, m.Mtyp)),
		Func:    toType(typeOffFor(tt, m.Tfn)),
		Index:   i,
	}
}

// MethodByName returns the method with that name in the type's method set
// and a boolean indicating if the method was found.
func (t Type) MethodByName(name string) (Method, bool) {
	if t.Kind() == Interface {
		tt := (*interfaceType)(unsafe.Pointer(t.common()))
		for i := range tt.Methods {
			m := &tt.Methods[i]
			if tt.nameOff(m.Name).Name() == name {
				return Method{
					Name:    tt.nameOff(m.Name).Name(),
					PkgPath: tt.nameOff(m.Name).PkgPath(),
					Type:    toType(tt.typeOff(m.Typ)),
					Func:    toType(tt.typeOff(m.Tfn)),
					Index:   i,
				}, true
			}
		}
		return Method{}, false
	}

	tt := t.common()
	ms := tt.ExportedMethods()
	for i := range ms {
		m := &ms[i]
		if nameOffFor(tt, m.Name).Name() == name {
			return Method{
				Name:    nameOffFor(tt, m.Name).Name(),
				PkgPath: nameOffFor(tt, m.Name).PkgPath(),
				Type:    toType(typeOffFor(tt, m.Mtyp)),
				Func:    toType(typeOffFor(tt, m.Tfn)),
				Index:   i,
			}, true
		}
	}
	return Method{}, false
}

// Note: Using reflect.Method and reflect.MethodByName ensures all public methods are retained and not eliminated by dead code elimination.

// Value.Method returns the i'th method in the value's method set.
// It panics if i is out of range or if the value is a nil interface value.
func (v Value) Method(i int) Value {
	if v.typ() == nil {
		panic(&ValueError{"reflect.Value.Method", Invalid})
	}
	if v.flag&flagMethod != 0 || uint(i) >= uint(toRType(v.typ()).NumMethod()) {
		panic("reflect: Method index out of range")
	}
	if v.typ().Kind() == Interface && v.IsNil() {
		panic("reflect: Method on nil interface value")
	}
	fl := v.flag.ro() | (v.flag & flagIndir)
	fl |= flag(Func)
	fl |= flag(i)<<flagMethodShift | flagMethod
	return Value{v.typ(), v.ptr, fl}
}

// Value.MethodByName returns the method with that name in the value's method set
// and a boolean indicating if the method was found.
func (v Value) MethodByName(name string) Value {
	if v.typ() == nil {
		panic(&ValueError{"reflect.Value.MethodByName", Invalid})
	}
	if v.flag&flagMethod != 0 {
		return Value{}
	}
	m, ok := toRType(v.typ()).MethodByName(name)
	if !ok {
		return Value{}
	}
	return v.Method(m.Index)
}

// Note: Using reflect.Value's Method and MethodByName ensures all public methods are retained and not eliminated by dead code elimination.
