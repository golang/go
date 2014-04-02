// +build ignore

package B1

import "time"

var startup = time.Now()

func example() time.Duration {
	before := time.Now()
	time.Sleep(1)
	return time.Now().Sub(before)
}

func msSinceStartup() int64 {
	return int64(time.Now().Sub(startup) / time.Millisecond)
}
