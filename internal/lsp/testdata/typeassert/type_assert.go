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
	*abcPtrImpl //@item(abcPtrImplPtr, "*abcPtrImpl", "struct{...}", "struct")

	var a abc
	switch a.(type) {
	case ab: //@complete(":", abcPtrImplPtr, abcImpl, abcIntf, abcNotImpl)
	case *ab: //@complete(":", abcImpl, abcPtrImpl, abcIntf, abcNotImpl)
	}

	a.(ab)  //@complete(")", abcPtrImplPtr, abcImpl, abcIntf, abcNotImpl)
	a.(*ab) //@complete(")", abcImpl, abcPtrImpl, abcIntf, abcNotImpl)
}
