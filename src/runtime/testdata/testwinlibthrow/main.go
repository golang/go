package main

import (
	"os"
	"syscall"
)

func main() {
	dll := syscall.MustLoadDLL("veh.dll")
	RaiseNoExcept := dll.MustFindProc("RaiseNoExcept")
	ThreadRaiseNoExcept := dll.MustFindProc("ThreadRaiseNoExcept")

	thread := len(os.Args) > 1 && os.Args[1] == "thread"
	if !thread {
		RaiseNoExcept.Call()
	} else {
		ThreadRaiseNoExcept.Call()
	}
}
