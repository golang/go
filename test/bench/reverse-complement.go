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
 */

package main

import (
	"bufio";
	"bytes";
	"os";
)

const	lineSize = 60

var complement = [256]uint8 {
	'A':	'T',	'a':	'T',
	'C':	'G',	'c':	'G',
	'G':	'C',	'g':	'C',
	'T':	'A',	't':	'A',
	'U':	'A',	'u':	'A',
	'M':	'K',	'm':	'K',
	'R':	'Y',	'r':	'Y',
	'W':	'W',	'w':	'W',
	'S':	'S',	's':	'S',
	'Y':	'R',	'y':	'R',
	'K':	'M',	'k':	'M',
	'V':	'B',	'v':	'B',
	'H':	'D',	'h':	'D',
	'D':	'H',	'd':	'H',
	'B':	'V',	'b':	'V',
	'N':	'N',	'n':	'N',
}

var in *bufio.Reader

func reverseComplement(in []byte) []byte {
	outLen := len(in) + (len(in) + lineSize -1)/lineSize;
	out := make([]byte, outLen);
	j := 0;
	k := 0;
	for i := len(in)-1; i >= 0; i-- {
		if k == lineSize {
			out[j] = '\n';
			j++;
			k = 0;
		}
		out[j] = complement[in[i]];
		j++;
		k++;
	}
	out[j] = '\n';
	j++;
	return out[0:j];
}

func output(buf []byte) {
	if len(buf) == 0 {
		return
	}
	os.Stdout.Write(reverseComplement(buf));
}

func main() {
	in = bufio.NewReader(os.Stdin);
	buf := make([]byte, 100*1024);
	top := 0;
	for {
		line, err := in.ReadLineSlice('\n');
		if err != nil {
			break
		}
		if line[0] == '>' {
			if top > 0 {
				output(buf[0:top]);
				top = 0;
			}
			os.Stdout.Write(line);
			continue
		}
		line = line[0:len(line)-1];	// drop newline
		if top+len(line) > len(buf) {
			nbuf := make([]byte, 2*len(buf) + 1024*(100+len(line)));
			bytes.Copy(nbuf, buf[0:top]);
			buf = nbuf;
		}
		bytes.Copy(buf[top:len(buf)], line);
		top += len(line);
	}
	output(buf[0:top]);
}
