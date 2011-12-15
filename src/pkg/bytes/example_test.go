package bytes_test

import (
	. "bytes"
	"encoding/base64"
	"io"
	"os"
)

// Hello world!
func ExampleBuffer() {
	var b Buffer // A Buffer needs no initialization.
	b.Write([]byte("Hello "))
	b.Write([]byte("world!"))
	b.WriteTo(os.Stdout)
}

// Gophers rule!
func ExampleBuffer_reader() {
	// A Buffer can turn a string or a []byte into an io.Reader.
	buf := NewBufferString("R29waGVycyBydWxlIQ==")
	dec := base64.NewDecoder(base64.StdEncoding, buf)
	io.Copy(os.Stdout, dec)
}
