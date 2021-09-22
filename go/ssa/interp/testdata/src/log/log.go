package log

import (
	"fmt"
	"os"
)

func Println(v ...interface{}) {
	fmt.Println(v...)
}
func Printf(format string, v ...interface{}) {
	fmt.Printf(format, v...)
}

func Fatalln(v ...interface{}) {
	Println(v...)
	os.Exit(1)
}

func Fatalf(format string, v ...interface{}) {
	Printf(format, v...)
	os.Exit(1)
}
