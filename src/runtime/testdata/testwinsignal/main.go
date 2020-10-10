package main

import (
	"net"
	"os"
	"os/signal"
	"strconv"
	"time"
)

func main() {
	c := make(chan os.Signal, 1)
	signal.Notify(c)

	con, _ := net.Dial("udp", os.Args[1])
	con.Write([]byte(strconv.Itoa(os.Getpid())))
	sig := <-c

	time.Sleep(time.Second)
	con.Write([]byte(sig.String()))
}
