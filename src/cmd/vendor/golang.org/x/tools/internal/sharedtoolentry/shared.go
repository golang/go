//go:build sharedtool

package sharedtoolentry

type exitStatus int

func (s exitStatus) SharedExitCode() int { return int(s) }

const enabled = true
