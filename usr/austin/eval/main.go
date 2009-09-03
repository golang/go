// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./_obj/eval";
	"bufio";
	"os";
)

func main() {
	w := eval.NewWorld();
	r := bufio.NewReader(os.Stdin);
	for {
		print("; ");
		line, err := r.ReadString('\n');
		if err != nil {
			break;
		}
		code, err := w.Compile(line);
		if err != nil {
			println(err.String());
			continue;
		}
		v, err := code.Run();
		if err != nil {
			println(err.String());
			continue;
		}
		if v != nil {
			println(v.String());
		}
	}
}

