package typeassert

type abc interface { //@item(abcIntf, "abc", "interface{...}", "interface")
	abc()
}

type abcImpl struct{} //@item(abcImpl, "abcImpl", "struct{...}", "struct")
func (abcImpl) abc()

type abcPtrImpl struct{} //@item(abcPtrImpl, "abcPtrImpl", "struct{...}", "struct")
func (*abcPtrImpl) abc()

type abcNotImpl struct{} //@item(abcNotImpl, "abcNotImpl", "struct{...}", "struct")

func _() {
	var a abc
	switch a.(type) {
	case ab: //@complete(":", abcImpl, abcIntf, abcNotImpl, abcPtrImpl)
	case *ab: //@complete(":", abcImpl, abcPtrImpl, abcIntf, abcNotImpl)
	}

	a.(ab)  //@complete(")", abcImpl, abcIntf, abcNotImpl, abcPtrImpl)
	a.(*ab) //@complete(")", abcImpl, abcPtrImpl, abcIntf, abcNotImpl)
}
