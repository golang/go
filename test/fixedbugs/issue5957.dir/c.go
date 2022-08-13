package p

import (
	"./a" // ERROR "imported and not used: \x22test/a\x22 as surprise|imported and not used: surprise"
	"./b" // ERROR "imported and not used: \x22test/b\x22 as surprise2|imported and not used: surprise2"
	b "./b" // ERROR "imported and not used: \x22test/b\x22$|imported and not used: surprise2"
	foo "math" // ERROR "imported and not used: \x22math\x22 as foo|imported and not used: math"
	"fmt" // actually used
	"strings" // ERROR "imported and not used: \x22strings\x22|imported and not used: strings"
)

var _ = fmt.Printf
