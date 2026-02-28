package a

import "log"

func Do() {
	Do2()
}

func Do2() {
	println(log.Ldate | log.Ltime | log.Lshortfile)
}
