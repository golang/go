// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package md5_test

import (
	"crypto/md5"
	"fmt"
	"io"
	"testing"
	"unsafe"
)

type md5Test struct {
	out string
	in  string
}

var golden = []md5Test{
	{"d41d8cd98f00b204e9800998ecf8427e", ""},
	{"0cc175b9c0f1b6a831c399e269772661", "a"},
	{"187ef4436122d1cc2f40dc2b92f0eba0", "ab"},
	{"900150983cd24fb0d6963f7d28e17f72", "abc"},
	{"e2fc714c4727ee9395f324cd2e7f331f", "abcd"},
	{"ab56b4d92b40713acc5af89985d4b786", "abcde"},
	{"e80b5017098950fc58aad83c8c14978e", "abcdef"},
	{"7ac66c0f148de9519b8bd264312c4d64", "abcdefg"},
	{"e8dc4081b13434b45189a720b77b6818", "abcdefgh"},
	{"8aa99b1f439ff71293e95357bac6fd94", "abcdefghi"},
	{"a925576942e94b2ef57a066101b48876", "abcdefghij"},
	{"d747fc1719c7eacb84058196cfe56d57", "Discard medicine more than two years old."},
	{"bff2dcb37ef3a44ba43ab144768ca837", "He who has a shady past knows that nice guys finish last."},
	{"0441015ecb54a7342d017ed1bcfdbea5", "I wouldn't marry him with a ten foot pole."},
	{"9e3cac8e9e9757a60c3ea391130d3689", "Free! Free!/A trip/to Mars/for 900/empty jars/Burma Shave"},
	{"a0f04459b031f916a59a35cc482dc039", "The days of the digital watch are numbered.  -Tom Stoppard"},
	{"e7a48e0fe884faf31475d2a04b1362cc", "Nepal premier won't resign."},
	{"637d2fe925c07c113800509964fb0e06", "For every action there is an equal and opposite government program."},
	{"834a8d18d5c6562119cf4c7f5086cb71", "His money is twice tainted: 'taint yours and 'taint mine."},
	{"de3a4d2fd6c73ec2db2abad23b444281", "There is no reason for any individual to have a computer in their home. -Ken Olsen, 1977"},
	{"acf203f997e2cf74ea3aff86985aefaf", "It's a tiny change to the code and not completely disgusting. - Bob Manchek"},
	{"e1c1384cb4d2221dfdd7c795a4222c9a", "size:  a.out:  bad magic"},
	{"c90f3ddecc54f34228c063d7525bf644", "The major problem is with sendmail.  -Mark Horton"},
	{"cdf7ab6c1fd49bd9933c43f3ea5af185", "Give me a rock, paper and scissors and I will move the world.  CCFestoon"},
	{"83bc85234942fc883c063cbd7f0ad5d0", "If the enemy is within range, then so are you."},
	{"277cbe255686b48dd7e8f389394d9299", "It's well we cannot hear the screams/That we create in others' dreams."},
	{"fd3fb0a7ffb8af16603f3d3af98f8e1f", "You remind me of a TV show, but that's all right: I watch it anyway."},
	{"469b13a78ebf297ecda64d4723655154", "C is as portable as Stonehedge!!"},
	{"63eb3a2f466410104731c4b037600110", "Even if I could be Shakespeare, I think I should still choose to be Faraday. - A. Huxley"},
	{"72c2ed7592debca1c90fc0100f931a2f", "The fugacity of a constituent in a mixture of gases at a given temperature is proportional to its mole fraction.  Lewis-Randall Rule"},
	{"132f7619d33b523b1d9e5bd8e0928355", "How can you write a big system without C++?  -Paul Glick"},
}

func TestGolden(t *testing.T) {
	for i := 0; i < len(golden); i++ {
		g := golden[i]
		c := md5.New()
		buf := make([]byte, len(g.in)+4)
		for j := 0; j < 3+4; j++ {
			if j < 2 {
				io.WriteString(c, g.in)
			} else if j == 2 {
				io.WriteString(c, g.in[0:len(g.in)/2])
				c.Sum(nil)
				io.WriteString(c, g.in[len(g.in)/2:])
			} else if j > 2 {
				// test unaligned write
				buf = buf[1:]
				copy(buf, g.in)
				c.Write(buf[:len(g.in)])
			}
			s := fmt.Sprintf("%x", c.Sum(nil))
			if s != g.out {
				t.Fatalf("md5[%d](%s) = %s want %s", j, g.in, s, g.out)
			}
			c.Reset()
		}
	}
}

func ExampleNew() {
	h := md5.New()
	io.WriteString(h, "The fog is getting thicker!")
	io.WriteString(h, "And Leon's getting laaarger!")
	fmt.Printf("%x", h.Sum(nil))
	// Output: e2c569be17396eca2a2e3c11578123ed
}

var bench = md5.New()
var buf = make([]byte, 8192+1)
var sum = make([]byte, bench.Size())

func benchmarkSize(b *testing.B, size int, unaligned bool) {
	b.SetBytes(int64(size))
	buf := buf
	if unaligned {
		if uintptr(unsafe.Pointer(&buf[0]))&(unsafe.Alignof(uint32(0))-1) == 0 {
			buf = buf[1:]
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bench.Reset()
		bench.Write(buf[:size])
		bench.Sum(sum[:0])
	}
}

func BenchmarkHash8Bytes(b *testing.B) {
	benchmarkSize(b, 8, false)
}

func BenchmarkHash1K(b *testing.B) {
	benchmarkSize(b, 1024, false)
}

func BenchmarkHash8K(b *testing.B) {
	benchmarkSize(b, 8192, false)
}

func BenchmarkHash8BytesUnaligned(b *testing.B) {
	benchmarkSize(b, 8, true)
}

func BenchmarkHash1KUnaligned(b *testing.B) {
	benchmarkSize(b, 1024, true)
}

func BenchmarkHash8KUnaligned(b *testing.B) {
	benchmarkSize(b, 8192, true)
}
