package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	// Ensure that this process terminates when the test times out,
	// even if the expected signal never arrives.
	go func() {
		io.Copy(io.Discard, os.Stdin)
		log.Fatal("stdin is closed; terminating")
	}()

	// Register to receive all signals.
	c := make(chan os.Signal, 1)
	signal.Notify(c)

	// Get console window handle.
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	getConsoleWindow := kernel32.NewProc("GetConsoleWindow")
	hwnd, _, err := getConsoleWindow.Call()
	if hwnd == 0 {
		log.Fatal("no associated console: ", err)
	}

	// Send message to close the console window.
	const _WM_CLOSE = 0x0010
	user32 := syscall.NewLazyDLL("user32.dll")
	postMessage := user32.NewProc("PostMessageW")
	ok, _, err := postMessage.Call(hwnd, _WM_CLOSE, 0, 0)
	if ok == 0 {
		log.Fatal("post message failed: ", err)
	}

	sig := <-c

	// Allow some time for the handler to complete if it's going to.
	//
	// (In https://go.dev/issue/41884 the handler returned immediately,
	// which caused Windows to terminate the program before the goroutine
	// that received the SIGTERM had a chance to actually clean up.)
	time.Sleep(time.Second)

	// Print the signal's name: "terminated" makes the test succeed.
	fmt.Println(sig)
}
