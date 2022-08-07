// run -race -gcflags=all="-N -l"
package main

import (
	"os/exec"
)

func main() {
	cmd := exec.Command("echo", "test")
	_ = cmd.Start()
}