package local

type Issue60575 struct{
	ID string
}

func f() {
	_ = &Issue60575{Id: "foo"} // ERROR "unknown field 'Id' in struct literal of type local.S (but does have ID)"
}
