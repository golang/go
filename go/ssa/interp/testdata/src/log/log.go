package log

import (
	"fmt"
	"os"
)

func Println(v ...interface{}) {
	fmt.Println(v...)
}

func Fatalln(v ...interface{}) {
	Println(v...)
	os.Exit(1)
}
