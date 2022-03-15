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
	// register to receive all signals
	c := make(chan os.Signal, 1)
	signal.Notify(c)

	// get console window handle
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	getConsoleWindow := kernel32.NewProc("GetConsoleWindow")
	hwnd, _, err := getConsoleWindow.Call()
	if hwnd == 0 {
		log.Fatal("no associated console: ", err)
	}

	// close console window
	const _WM_CLOSE = 0x0010
	user32 := syscall.NewLazyDLL("user32.dll")
	postMessage := user32.NewProc("PostMessageW")
	ok, _, err := postMessage.Call(hwnd, _WM_CLOSE, 0, 0)
	if ok == 0 {
		log.Fatal("post message failed: ", err)
	}

	// check if we receive syscall.SIGTERM
	select {
	case sig := <-c:
		// prentend to take some time handling the signal
		time.Sleep(time.Second)
		// print signal name, "terminated" makes the test succeed
		fmt.Println(sig)

	case <-time.After(time.Second):
		log.Fatal("timed out waiting for signal")
	}
}
