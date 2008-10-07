// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $F.go && $L $F.$A && GOMAXPROCS=3 ./$A.out
// # TODO(rsc): GOMAXPROCS will go away eventually.
// # 3 is one for Echo, one for Serve, one for Connect.

package main
import (
	"os";
	"io";
	"net";
	"syscall"
)

func StringToBuf(s string) *[]byte  {
	l := len(s);
	b := new([]byte, l);
	for i := 0; i < l; i++ {
		b[i] = s[i];
	}
	return b;
}

func Echo(fd io.ReadWrite, done *chan<- int) {
	var buf [1024]byte;

	for {
		n, err := fd.Read(&buf);
		if err != nil || n == 0 {
			break;
		}
		fd.Write((&buf)[0:n])
	}
	done <- 1
}

func Serve(network, addr string, listening, done *chan<- int) {
	l, err := net.Listen(network, addr);
	if err != nil {
		panic("listen: "+err.String());
	}
	listening <- 1;

	for {
		fd, addr, err := l.Accept();
		if err != nil {
			break;
		}
		echodone := new(chan int);
		go Echo(fd, echodone);
		<-echodone;	// make sure Echo stops
		l.Close();
	}
	done <- 1
}

func Connect(network, addr string) {
	fd, err := net.Dial(network, "", addr);
	if err != nil {
		panic("connect: "+err.String());
	}

	b := StringToBuf("hello, world\n");
	var b1 [100]byte;

	n, errno := fd.Write(b);
	if n != len(b) {
		panic("syscall.write in connect");
	}

	n, errno = fd.Read(&b1);
	if n != len(b) {
		panic("syscall.read in connect");
	}

//	os.Stdout.Write((&b1)[0:n]);
	fd.Close();
}

func Test(network, listenaddr, dialaddr string) {
//	print("Test ", network, " ", listenaddr, " ", dialaddr, "\n");
	listening := new(chan int);
	done := new(chan int);
	go Serve(network, listenaddr, listening, done);
	<-listening;	// wait for server to start
	Connect(network, dialaddr);
	<-done;	// make sure server stopped
}

func main() {
	Test("tcp", "0.0.0.0:9999", "127.0.0.1:9999");
	Test("tcp", "[::]:9999", "[::ffff:127.0.0.1]:9999");
	Test("tcp", "[::]:9999", "127.0.0.1:9999");
	Test("tcp", "0.0.0.0:9999", "[::ffff:127.0.0.1]:9999");
	sys.exit(0);	// supposed to happen on return, doesn't
}

