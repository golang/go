package main

import (
	"fmt"
	"os"
	"runtime"

	"golang.org/x/net/dns/dnsmessage"
)

func main() {

	name := []byte(os.Args[1])

	var buffer = [512]byte{}
	ret := runtime.Res_search(&name[0], 1, 1, &buffer[0], 512)
	if ret != 0 {
		fmt.Println(ret)
	}

	bufferSlice := buffer[:]

	msg := &dnsmessage.Message{}

	err := msg.Unpack(bufferSlice)
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println("\n\n\n\n")
	fmt.Println(msg.Answers[0].Body)
}
