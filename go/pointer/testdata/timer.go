// +build ignore

package main

import "time"

func after() {}

func main() {
	// @calls time.startTimer -> time.sendTime
	ticker := time.NewTicker(1)
	<-ticker.C

	// @calls time.startTimer -> time.sendTime
	timer := time.NewTimer(time.Second)
	<-timer.C

	// @calls time.startTimer -> time.goFunc
	// @calls time.goFunc -> main.after
	timer = time.AfterFunc(time.Second, after)
	<-timer.C
}

// @calls time.sendTime -> time.Now
