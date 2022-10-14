package main

import (
	"os"
	"syscall"
)

func main() {
	dll := syscall.MustLoadDLL("veh.dll")
	RaiseExcept := dll.MustFindProc("RaiseExcept")
	RaiseNoExcept := dll.MustFindProc("RaiseNoExcept")
	ThreadRaiseExcept := dll.MustFindProc("ThreadRaiseExcept")
	ThreadRaiseNoExcept := dll.MustFindProc("ThreadRaiseNoExcept")

	thread := len(os.Args) > 1 && os.Args[1] == "thread"
	if !thread {
		RaiseExcept.Call()
		RaiseNoExcept.Call()
	} else {
		ThreadRaiseExcept.Call()
		ThreadRaiseNoExcept.Call()
	}
}
