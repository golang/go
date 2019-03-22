package fmt

func Sprint(args ...interface{}) string

func Print(args ...interface{}) {
	for i, arg := range args {
		if i > 0 {
			print(" ")
		}
		print(Sprint(arg))
	}
}

func Println(args ...interface{}) {
	Print(args...)
	println()
}

// formatting is too complex to fake

func Printf(args ...interface{}) string {
	panic("Printf is not supported")
}
func Sprintf(format string, args ...interface{}) string {
	panic("Sprintf is not supported")
}
