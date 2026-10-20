// Violator child (M20-S1): a C function that calls back into Go,
// once through the directive pair (fast path) and once through
// nocallback alone (plain path). Each violation panics with the
// documented message; a second violation on the same goroutine
// reports the stuck flag. The program exits via os.Exit so the
// burned callback flag never meets the goroutine-exit check.
package main

/*
extern int GoCallback(int v);

#cgo noescape vfast
#cgo nocallback vfast
int vfast(void) { return GoCallback(41); }

#cgo nocallback vplain
int vplain(void) { return GoCallback(7); }

int vadd(int a, int b) { return a + b; }
*/
import "C"

import (
	"fmt"
	"os"
)

func violate(fn func() int32) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg, _ = r.(string)
		}
	}()
	fn()
	return "returned"
}

func main() {
	var fn func() int32
	if len(os.Args) > 1 && os.Args[1] == "plain" {
		fn = func() int32 { return int32(C.vplain()) }
	} else {
		fn = func() int32 { return int32(C.vfast()) }
	}
	fmt.Println("first:", violate(fn))
	fmt.Println("second:", violate(fn))
	fmt.Println("add:", int(C.vadd(20, 22)))
	os.Exit(0)
}
