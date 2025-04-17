package p

import (
	"./a" // ERROR "imported and not used: \x22test/a\x22 as surprise|imported and not used: surprise|\x22test/a\x22 imported as surprise and not used"
	"./b" // ERROR "imported and not used: \x22test/b\x22 as surprise2|imported and not used: surprise2|\x22test/b\x22 imported as surprise2 and not used"
	b "./b" // ERROR "imported and not used: \x22test/b\x22$|imported and not used: surprise2|\x22test/b\x22 imported and not used"
	foo "math" // ERROR "imported and not used: \x22math\x22 as foo|imported and not used: math|\x22math\x22 imported as foo and not used"
	"fmt" // actually used
	"strings" // ERROR "imported and not used: \x22strings\x22|imported and not used: strings|\x22strings\x22 imported and not used"
)

var _ = fmt.Printf
