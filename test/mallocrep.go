// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Repeated malloc test.

package main

import (
	"flag";
	"malloc"
)

var chatty bool;
var chatty_flag = flag.Bool("v", false, &chatty, "chatty");

var oldsys uint64;
func bigger() {
	if st := malloc.GetStats(); oldsys < st.sys {
		oldsys = st.sys;
		if chatty {
			println(st.sys, " system bytes for ", st.alloc, " Go bytes");
		}
		if st.sys > 1e9 {
			panicln("too big");
		}
	}
}

func main() {
	flag.Parse();
	for i := 0; i < 1<<8; i++ {
		for j := 1; j <= 1<<22; j<<=1 {
			if i == 0 && chatty {
				println("First alloc:", j);
			}
			b := malloc.Alloc(uint64(j));
			malloc.Free(b);
			if a := malloc.GetStats().alloc; a != 0 {
				panicln("malloc wrong count", a);
			}
			bigger();
		}
		if i%(1<<10) == 0 && chatty {
			println(i);
		}
		if i == 0 {
			if chatty {
				println("Primed", i);
			}
		//	malloc.frozen = true;
		}
	}
}
