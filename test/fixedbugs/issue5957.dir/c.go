package p

import (
	"./a" // ERROR "imported and not used: \x22a\x22 as surprise"
	"./b" // ERROR "imported and not used: \x22b\x22 as surprise2"
	b "./b" // ERROR "imported and not used: \x22b\x22$"
	foo "math" // ERROR "imported and not used: \x22math\x22 as foo"
	"fmt" // actually used
	"strings" // ERROR "imported and not used: \x22strings\x22"
)

var _ = fmt.Printf
