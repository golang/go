package issue30628

import (
	"os"
	"sync"
)

const numR = int32(os.O_TRUNC + 5)

type Apple struct {
	hey sync.RWMutex
	x   int
	RQ  [numR]struct {
		Count    uintptr
		NumBytes uintptr
		Last     uintptr
	}
}
