/*
 * Project: grpc-go
 * Issue or PR  : https://github.com/grpc/grpc-go/pull/1275
 * Buggy version: (missing)
 * fix commit-id: 0669f3f89e0330e94bb13fa1ce8cc704aab50c9c
 * Flaky: 100/100
 * Description:
 *   Two goroutines are invovled in this deadlock. The first goroutine
 * is the main goroutine. It is blocked at case <- donec, and it is
 * waiting for the second goroutine to close the channel.
 *   The second goroutine is created by the main goroutine. It is blocked
 * when calling stream.Read(). stream.Read() invokes recvBufferRead.Read().
 * The second goroutine is blocked at case i := r.recv.get(), and it is
 * waiting for someone to send a message to this channel.
 *   It is the client.CloseSream() method called by the main goroutine that
 * should send the message, but it is not. The patch is to send out this message.
 */
package main

import (
	"io"
	"runtime"
	"time"
)

func init() {
	register("Grpc1275", Grpc1275)
}

type recvBuffer_grpc1275 struct {
	c chan bool
}

func (b *recvBuffer_grpc1275) get() <-chan bool {
	return b.c
}

type recvBufferReader_grpc1275 struct {
	recv *recvBuffer_grpc1275
}

func (r *recvBufferReader_grpc1275) Read(p []byte) (int, error) {
	select {
	case <-r.recv.get():
	}
	return 0, nil
}

type Stream_grpc1275 struct {
	trReader io.Reader
}

func (s *Stream_grpc1275) Read(p []byte) (int, error) {
	return io.ReadFull(s.trReader, p)
}

type http2Client_grpc1275 struct{}

func (t *http2Client_grpc1275) CloseStream(s *Stream_grpc1275) {
	// It is the client.CloseSream() method called by the
	// main goroutine that should send the message, but it
	// is not. The patch is to send out this message.
}

func (t *http2Client_grpc1275) NewStream() *Stream_grpc1275 {
	return &Stream_grpc1275{
		trReader: &recvBufferReader_grpc1275{
			recv: &recvBuffer_grpc1275{
				c: make(chan bool),
			},
		},
	}
}

func testInflightStreamClosing_grpc1275() {
	client := &http2Client_grpc1275{}
	stream := client.NewStream()
	donec := make(chan bool)
	go func() { // G2
		defer close(donec)
		// deadlocks: 1
		stream.Read([]byte{1})
	}()

	client.CloseStream(stream)

	timeout := time.NewTimer(300 * time.Nanosecond)
	select {
	case <-donec:
		if !timeout.Stop() {
			<-timeout.C
		}
	case <-timeout.C:
	}
}

///
/// G1 									G2
/// testInflightStreamClosing()
/// 									stream.Read()
/// 									io.ReadFull()
/// 									<- r.recv.get()
/// CloseStream()
/// <- donec
/// ------------G1 timeout, G2 leak---------------------
///

func Grpc1275() {
	defer func() {
		time.Sleep(100 * time.Millisecond)
		runtime.GC()
	}()
	go func() {
		testInflightStreamClosing_grpc1275() // G1
	}()
}
