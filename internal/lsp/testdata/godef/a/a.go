// A comment just to push the positions out

package a

import "fmt"

type A string //@A

func Stuff() { //@Stuff
	x := 5
	Random2(x) //@godef("dom2", Random2)
	Random()   //@godef("()", Random)

	var err error         //@err
	fmt.Printf("%v", err) //@godef("err", err)
}
