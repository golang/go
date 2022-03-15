package main

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	c := make(chan os.Signal, 1)
	signal.Notify(c)

	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	getConsoleWindow := kernel32.NewProc("GetConsoleWindow")
	hwnd, _, err := getConsoleWindow.Call()
	if hwnd == 0 {
		log.Fatal("no associated console: ", err)
	}

	const _WM_CLOSE = 0x0010
	user32 := syscall.NewLazyDLL("user32.dll")
	postMessage := user32.NewProc("PostMessageW")
	ok, _, err := postMessage.Call(hwnd, _WM_CLOSE, 0, 0)
	if ok == 0 {
		log.Fatal("post message failed: ", err)
	}

	sig := <-c

	time.Sleep(time.Second)
	fmt.Println(sig)
}
