//go:build gofuzz

package main

import (
	"fmt"
	"os"
	"runtime/debug"
	"syscall"
	"unsafe"
)

// #include <stdint.h>
import "C"

//export LLVMFuzzerTestOneInput
func LLVMFuzzerTestOneInput(data *C.char, size C.size_t) C.int {
	s := C.GoBytes(unsafe.Pointer(data), C.int(size))
	defer catchPanics()
	harness(s)
	return 0
}

func catchPanics() {
	if r := recover(); r != nil {
		// print panic information
		fmt.Printf("Go panic: %v\n", r)
		debug.PrintStack()
		syscall.Kill(os.Getpid(), syscall.SIGABRT)
	}
}

//export LLVMFuzzerInitialize
func LLVMFuzzerInitialize(argc *C.int, argv ***C.char) C.int {
	return 0
}

func main() {
	if len(os.Args) == 2 {
		path := os.Args[1]
		info, err := os.Stat(path)
		if err != nil {
			fmt.Println("Failed to access path", err)
		}

		if info.IsDir() {
			files, _ := os.ReadDir(path)
			for _, file := range files {
				filePath := path + string(os.PathSeparator) + file.Name()
				content, _ := os.ReadFile(filePath)
				cSize := C.size_t(len(content))
				if cSize > 1 {
					cData := (*C.char)(unsafe.Pointer(&content[0]))
					LLVMFuzzerTestOneInput(cData, cSize)
				}
			}
		} else {
			content, _ := os.ReadFile(path)
			cSize := C.size_t(len(content))
			if cSize > 1 {
				cData := (*C.char)(unsafe.Pointer(&content[0]))
				LLVMFuzzerTestOneInput(cData, cSize)
			}
		}
	} else {
		fmt.Println("Usage: <folderPath>")
	}
}
