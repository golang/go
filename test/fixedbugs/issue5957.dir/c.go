package p

import (
	"./a"      // ERROR "imported but not used: \x22a\x22 as surprise|imported but not used: surprise"
	"./b"      // ERROR "imported but not used: \x22b\x22 as surprise2|imported but not used: surprise2"
	b "./b"    // ERROR "imported but not used: \x22b\x22$|imported but not used: surprise2"
	"fmt"      // actually used
	foo "math" // ERROR "imported but not used: \x22math\x22 as foo|imported but not used: math"
	"strings"  // ERROR "imported but not used: \x22strings\x22|imported but not used: strings"
)

var _ = fmt.Printf
