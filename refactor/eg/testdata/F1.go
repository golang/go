// +build ignore

package F1

import "sync"

func example(n int) {
	var x struct {
		mutex sync.RWMutex
	}

	var y struct {
		sync.RWMutex
	}

	type l struct {
		sync.RWMutex
	}

	var z struct {
		l
	}

	var a struct {
		*l
	}

	var b struct{ Lock func() }

	// Match
	x.mutex.Lock()

	// Match
	y.Lock()

	// Match indirect
	z.Lock()

	// Should be no match however currently matches due to:
	// https://golang.org/issue/8584
	// Will start failing when this is fixed then just change golden to
	// No match pointer indirect
	// a.Lock()
	a.Lock()

	// No match
	b.Lock()
}
