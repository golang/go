package ir

// goLocalInitMap is mapping need init virtual variable to its go_local variable.
var goLocalInitMap = map[*Name]*Name{}

// BindGoLocalInit records need init virtual variable and its go_local variable.
func BindGoLocalInit(goLocal, needInit *Name) {
	goLocalInitMap[needInit] = goLocal
}

// GetGoLocalByInit gets go_local variable for need init virtual variable.
func GetGoLocalByInit(needInit *Name) *Name {
	return goLocalInitMap[needInit]
}
