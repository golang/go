package main

/*
extern int GoCallback(int v);

static int cd(int v, int n) {
	volatile char pad[512];
	pad[0] = (char)v;
	if (n <= 0) {
		return GoCallback(v);
	}
	return cd(v, n-1) + pad[0] - (char)v;
}

#cgo noescape vfast
#cgo nocallback vfast
int vfast(void) { return cd(41, 48); }

#cgo noescape cadd
#cgo nocallback cadd
int cadd(int a, int b) { return a + b; }

#cgo nocallback vplain
int vplain(void) { return GoCallback(7); }

int vadd(int a, int b) { return a + b; }
*/
import "C"

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
)

const violationRepeats = 1 << 18

func violate(fn func() int32) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg, _ = r.(string)
		}
	}()
	fn()
	return "returned"
}

func recoverResult(fn func() int32) (res int32, msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg, _ = r.(string)
		}
	}()
	return fn(), ""
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "plain" {
		fn := func() int32 { return int32(C.vplain()) }
		fmt.Println("first:", violate(fn))
		fmt.Println("second:", violate(fn))
		fmt.Println("add:", int(C.vadd(20, 22)))
		os.Exit(0)
	}
	fn := func() int32 { return int32(C.vfast()) }
	fmt.Println("first:", violate(fn))
	ok := 0
	for range violationRepeats {
		if strings.Contains(violate(fn), "called back into Go") {
			ok++
		}
	}
	fmt.Printf("loop: %d/%d\n", ok, violationRepeats)
	if ok == violationRepeats {
		var exiting sync.WaitGroup
		for range 64 {
			exiting.Go(func() { violate(fn) })
		}
		exiting.Wait()
		runtime.Gosched()
	}
	add, stuck := recoverResult(func() int32 { return int32(C.cadd(20, 22)) })
	if stuck != "" {
		fmt.Println("add:", stuck)
	} else {
		fmt.Println("add:", add)
	}
	os.Exit(0)
}
