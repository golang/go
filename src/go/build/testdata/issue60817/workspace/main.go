package main

import "workspace/with_generics"

func main() {

	_ = with_generics.T1[any]{
		val: 0,
	}
}
