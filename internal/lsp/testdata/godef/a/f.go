package a

import "fmt"

func TypeStuff() { //@Stuff
	var x string

	switch y := interface{}(x).(type) { //@mark(switchY, "y"),mark(switchStringY,"y"),godef("y", switchY)
	case int:
		fmt.Printf("%v", y) //@godef("y", switchY)
	case string:
		fmt.Printf("%v", y) //@godef("y", switchStringY)
	}

}
