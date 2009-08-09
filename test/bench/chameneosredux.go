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
	"flag";
	"fmt";
	"os";
)

var n = flag.Int("n", 600, "count")

type Color int

const (
	blue Color = iota;
	red;
	yellow;
)

func (c Color) String() string {
	return []string{"blue", "red", "yellow"}[c]
}

func complement(c1, c2 Color) Color {
	 switch c1 << 2 | c2 {
	 case blue << 2 | blue:
	 	return blue
	 case blue << 2 | red:
	 	return yellow
	 case blue << 2 | yellow:
	 	return red
	 case red << 2 | blue:
	 	return yellow
	 case red << 2 | red:
	 	return red
	 case red << 2 | yellow:
	 	return blue
	 case yellow << 2 | blue:
	 	return red
	 case yellow << 2 | red:
	 	return blue
	 case yellow << 2 | yellow:
	 	return yellow
	}
	fmt.Println("invalid colors", c1, c2);
	os.Exit(2);
	return 0
}

func printColors(c1, c2 Color) {
	fmt.Printf("%s + %s -> %s\n", c1, c2, complement(c1, c2));
}

func printColorTable() {
	printColors(blue, blue);
	printColors(blue, red);
	printColors(blue, yellow);
	printColors(red, blue);
	printColors(red, red);
	printColors(red, yellow);
	printColors(yellow, blue);
	printColors(yellow, red);
	printColors(yellow, yellow);
}

type Referee struct {
	rendezCount	int;
	cham	[]*Chameneos;
	rendez	chan *Chameneos;
	done	chan int;
}

func NewReferee() *Referee {
	ref := new(Referee);
	ref.cham = make([]*Chameneos, 0, 100);
	ref.rendez = make(chan *Chameneos);
	ref.done = make(chan int);
	go ref.Serve();
	return ref;
}

func (ref *Referee) Serve() {
	for i := 0; i < *n; i++ {
		c1 := <-ref.rendez;
		c2 := <-ref.rendez;
		c1.col, c2.col = complement(c1.col, c2.col), complement(c2.col, c1.col);
		c1.rendez <- c2;
		c2.rendez <- c1;
	}
	for i := 0; i < len(ref.cham); i++ {
		c := <-ref.rendez;
		c.rendez <- nil;
	}
	ref.done <- 1;
}

func (ref *Referee) Add(ch *Chameneos) {
	n := len(ref.cham);
	ref.cham = ref.cham[0:n+1];
	ref.cham[n] = ch;
}

type Chameneos struct {
	index	int;
	col	Color;
	rendez	chan *Chameneos;
	count	int;
	same	int;
	ref	*Referee;
}

func (c *Chameneos) Init(index int, ref *Referee, col Color) *Chameneos {
	c.index = index;
	c.ref = ref;
	c.col = col;
	c.rendez = make(chan *Chameneos);
	go c.Serve();
	return c;
}

func (c *Chameneos) Serve() {
	for {
		c.ref.rendez <- c;
		c1 := <- c.rendez;
		if c1 == nil {
			break
		}
		if c1.index == c.index {
			c.same++
		}
		c.count++;
	}
}

func play(ref *Referee, color []Color) {
	cham := make([]Chameneos, len(color));
	for i, c := range color {
		fmt.Printf(" %s", c);
		ref.Add(cham[i].Init(i, ref, c));
	}
	fmt.Printf("\n");
	<-ref.done;
	total := 0;
	for i, c := range cham {
		total += c.count;
		fmt.Printf("%d %s\n", c.count, say(c.same));
	}
	fmt.Printf("%s\n", say(total));
}

var words = []string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}

func say(n int) string {
	digits := fmt.Sprint(n);
	s := "";
	for _, c := range digits {
		s += " " + words[c-'0'];
	}
	return s;
}

func main() {
	flag.Parse();
	printColorTable();
	fmt.Print("\n");
	play(NewReferee(), []Color{blue, red, yellow});
	fmt.Print("\n");
	play(NewReferee(), []Color{blue, red, yellow, red, yellow, blue, red, yellow, red, blue});
}
