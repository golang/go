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
	"fmt"
	"io/ioutil"
	"os"
	"runtime"
	"regexp"
)

var variants = []string{
	"agggtaaa|tttaccct",
	"[cgt]gggtaaa|tttaccc[acg]",
	"a[act]ggtaaa|tttacc[agt]t",
	"ag[act]gtaaa|tttac[agt]ct",
	"agg[act]taaa|ttta[agt]cct",
	"aggg[acg]aaa|ttt[cgt]ccct",
	"agggt[cgt]aa|tt[acg]accct",
	"agggta[cgt]a|t[acg]taccct",
	"agggtaa[cgt]|[acg]ttaccct",
}

type Subst struct {
	pat, repl string
}

var substs = []Subst{
	Subst{"B", "(c|g|t)"},
	Subst{"D", "(a|g|t)"},
	Subst{"H", "(a|c|t)"},
	Subst{"K", "(g|t)"},
	Subst{"M", "(a|c)"},
	Subst{"N", "(a|c|g|t)"},
	Subst{"R", "(a|g)"},
	Subst{"S", "(c|g)"},
	Subst{"V", "(a|c|g)"},
	Subst{"W", "(a|t)"},
	Subst{"Y", "(c|t)"},
}

func countMatches(pat string, bytes []byte) int {
	re := regexp.MustCompile(pat)
	n := 0
	for {
		e := re.FindIndex(bytes)
		if e == nil {
			break
		}
		n++
		bytes = bytes[e[1]:]
	}
	return n
}

func main() {
	runtime.GOMAXPROCS(4)
	bytes, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintf(os.Stderr, "can't read input: %s\n", err)
		os.Exit(2)
	}
	ilen := len(bytes)
	// Delete the comment lines and newlines
	bytes = regexp.MustCompile("(>[^\n]+)?\n").ReplaceAll(bytes, []byte{})
	clen := len(bytes)

	mresults := make([]chan int, len(variants))
	for i, s := range variants {
		ch := make(chan int)
		mresults[i] = ch
		go func(ss string) {
			ch <- countMatches(ss, bytes)
		}(s)
	}

	lenresult := make(chan int)
	bb := bytes
	go func() {
		for _, sub := range substs {
			bb = regexp.MustCompile(sub.pat).ReplaceAll(bb, []byte(sub.repl))
		}
		lenresult <- len(bb)
	}()

	for i, s := range variants {
		fmt.Printf("%s %d\n", s, <-mresults[i])
	}
	fmt.Printf("\n%d\n%d\n%d\n", ilen, clen, <-lenresult)
}
