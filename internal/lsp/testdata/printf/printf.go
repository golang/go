package printf

import "fmt"

func myPrintf(string, ...interface{}) {}

func _() {
	var (
		aInt      int          //@item(printfInt, "aInt", "int", "var")
		aFloat    float64      //@item(printfFloat, "aFloat", "float64", "var")
		aString   string       //@item(printfString, "aString", "string", "var")
		aBytes    []byte       //@item(printfBytes, "aBytes", "[]byte", "var")
		aStringer fmt.Stringer //@item(printfStringer, "aStringer", "fmt.Stringer", "var")
		aError    error        //@item(printfError, "aError", "error", "var")
		aBool     bool         //@item(printfBool, "aBool", "bool", "var")
	)

	myPrintf("%d", a)       //@rank(")", printfInt, printfFloat)
	myPrintf("%s", a)       //@rank(")", printfString, printfInt),rank(")", printfBytes, printfInt),rank(")", printfStringer, printfInt),rank(")", printfError, printfInt)
	myPrintf("%w", a)       //@rank(")", printfError, printfInt)
	myPrintf("%x %[1]b", a) //@rank(")", printfInt, printfString)

	fmt.Printf("%t", a) //@rank(")", printfBool, printfInt)

	fmt.Fprintf(nil, "%f", a) //@rank(")", printfFloat, printfInt)

	fmt.Sprintf("%[2]q %[1]*.[3]*[4]f",
		a, //@rank(",", printfInt, printfFloat)
		a, //@rank(",", printfString, printfFloat)
		a, //@rank(",", printfInt, printfFloat)
		a, //@rank(",", printfFloat, printfInt)
	)
}
