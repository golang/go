package a

import "fmt"

func TypeStuff() { //@Stuff
	var x string

	switch y := interface{}(x).(type) { //@mark(switchY, "y"),godef("y", switchY)
	case int: //@mark(intY, "int")
		fmt.Printf("%v", y) //@hover("y", intY)
	case string: //@mark(stringY, "string")
		fmt.Printf("%v", y) //@hover("y", stringY)
	}

}
