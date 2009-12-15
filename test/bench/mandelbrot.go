/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    * Neither the name of "The Computer Language Benchmarks Game" nor the
    name of "The Computer Language Shootout Benchmarks" nor the names of
    its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/* The Computer Language Benchmarks Game
 * http://shootout.alioth.debian.org/
 *
 * contributed by The Go Authors.
 * Based on mandelbrot.c contributed by Greg Buchholz
 */

package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
)

var n = flag.Int("n", 200, "size")

func main() {
	flag.Parse()
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	w := *n
	h := *n
	bit_num := 0
	byte_acc := byte(0)
	const Iter = 50
	const Zero float64 = 0
	const Limit = 2.0

	fmt.Fprintf(out, "P4\n%d %d\n", w, h)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			Zr, Zi, Tr, Ti := Zero, Zero, Zero, Zero
			Cr := (2*float64(x)/float64(w) - 1.5)
			Ci := (2*float64(y)/float64(h) - 1.0)

			for i := 0; i < Iter && (Tr+Ti <= Limit*Limit); i++ {
				Zi = 2*Zr*Zi + Ci
				Zr = Tr - Ti + Cr
				Tr = Zr * Zr
				Ti = Zi * Zi
			}

			byte_acc <<= 1
			if Tr+Ti <= Limit*Limit {
				byte_acc |= 0x01
			}

			bit_num++

			if bit_num == 8 {
				out.WriteByte(byte_acc)
				byte_acc = 0
				bit_num = 0
			} else if x == w-1 {
				byte_acc <<= uint(8 - w%8)
				out.WriteByte(byte_acc)
				byte_acc = 0
				bit_num = 0
			}
		}
	}
}
