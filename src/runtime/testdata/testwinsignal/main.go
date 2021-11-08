package main

import (
	"fmt"
	"os"
	"os/signal"
	"time"
)

func main() {
	c := make(chan os.Signal, 1)
	signal.Notify(c)

	fmt.Println("ready")
	sig := <-c

	time.Sleep(time.Second)
	fmt.Println(sig)
}
