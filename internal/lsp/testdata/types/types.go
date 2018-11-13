package types

type X struct { //@item(X_struct, "X", "struct{...}", "struct")
	x int
}

type Y struct { //@item(Y_struct, "Y", "struct{...}", "struct")
	y int
}

type Bob interface { //@item(Bob_interface, "Bob", "interface{...}", "interface")
	Bobby()
}

func (*X) Bobby() {}
func (*Y) Bobby() {}
