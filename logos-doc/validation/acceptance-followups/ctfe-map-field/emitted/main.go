// run

package main

//line main.lgo:5:1
import "fmt"

//line main.lgo:7:1
type S struct{ M map[string]int }

func storeField() int {
//line main.lgo:10:1
	s := S{M: map[string]int{}}
//line main.lgo:11:1
	s.M["k"] = 4
//line main.lgo:12:1
	return func() int {
		if _logos_v, _logos_ok := s.M["k"]; _logos_ok {
			return _logos_v
		}
		return -1
	}()
//line main.lgo:13:1
}

//line main.lgo:15:1
const stored = 4

// The array assignment requires stored to be the compile-time constant 4.
//
//line main.lgo:18:1
var _ [4]struct{} = [stored]struct{}{}

//line main.lgo:20:1
func main() {
//line main.lgo:21:1
	if got := storeField(); got != stored {
//line main.lgo:22:1
		panic("map field store differs between compile time and runtime")
//line main.lgo:23:1
	}
//line main.lgo:24:1
	fmt.Println(stored)
//line main.lgo:25:1
}
