// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos

package unix

import (
	"sync"
)

// Single-stepping emulation for z/OS
//
// z/OS does not have native single-step support like Linux's PTRACE_SINGLESTEP.
// Instead, single-stepping must be emulated using temporary breakpoints.
//
// This file provides a complete implementation of single-step emulation using
// temporary breakpoints at both the sequential next instruction and branch target.

const (
	// NO_BRANCH_DEST indicates the instruction is not a branch or cannot branch
	NO_BRANCH_DEST = 0xFFFFFFFF

	// SVC_144 is the breakpoint instruction (SVC 144 = 0x0A90)
	SVC_144 = 0x0A90

	// Maximum number of temporary breakpoints (one for next PC, one for branch dest)
	MAX_TEMP_BREAKPOINTS = 2

	// S/390 branch instruction opcodes
	// RR format (Register-Register)
	opBCR   = 0x07 // Branch on Condition Register
	opBALR  = 0x05 // Branch and Link Register
	opBASR  = 0x0D // Branch and Save Register
	opBASSM = 0x0C // Branch and Save and Set Mode
	opBSM   = 0x0B // Branch and Set Mode
	opBCTR  = 0x06 // Branch on Count Register

	// RX format (Register-Index-Storage)
	opBC  = 0x47 // Branch on Condition
	opBAL = 0x45 // Branch and Link
	opBCT = 0x46 // Branch on Count
	opBAS = 0x4D // Branch and Save

	// RS format (Register-Storage)
	opBXH  = 0x86 // Branch on Index High
	opBXLE = 0x87 // Branch on Index Low or Equal

	// Extended branch opcodes
	opBA7 = 0xA7 // A7x series (BRC, BRAS, BRCT, BRCTG)
	opBC0 = 0xC0 // C0x series (BRASL, BRCL)
	opBEC = 0xEC // ECxx series (RIEc format: compare-and-branch)
)

// tempBreakpoint represents a temporary breakpoint for single-stepping
type tempBreakpoint struct {
	address      uint64
	originalInsn [8]byte // Save up to 8 bytes (full instruction)
	active       bool
}

// singleStepState manages temporary breakpoints for a process
type singleStepState struct {
	mu          sync.Mutex
	breakpoints [MAX_TEMP_BREAKPOINTS]tempBreakpoint
}

var (
	// Global map of process states (pid -> state)
	processStates   = make(map[int]*singleStepState)
	processStatesMu sync.Mutex
)

// getProcessState returns or creates the single-step state for a process
func getProcessState(pid int) *singleStepState {
	processStatesMu.Lock()
	defer processStatesMu.Unlock()

	state, exists := processStates[pid]
	if !exists {
		state = &singleStepState{}
		processStates[pid] = state
	}
	return state
}

// cleanupProcessState removes the state for a process (call on detach)
func cleanupProcessState(pid int) {
	processStatesMu.Lock()
	defer processStatesMu.Unlock()
	delete(processStates, pid)
}

// GetInstructionLength returns the length of an S/390 instruction in bytes.
func GetInstructionLength(firstByte byte) int {
	switch (firstByte >> 6) & 0x03 {
	case 0x00:
		return 2
	case 0x01, 0x02:
		return 4
	case 0x03:
		return 6
	default:
		return 2
	}
}

// GetNextInstructionAddr returns the address of the next sequential instruction.
func GetNextInstructionAddr(pc uint64, instruction []byte) uint64 {
	if len(instruction) < 1 {
		return pc + 2
	}
	insnLen := GetInstructionLength(instruction[0])
	return pc + uint64(insnLen)
}

// CalculateBranchDest calculates the destination address for branch instructions.
func CalculateBranchDest(pid int, pc uint64, instruction []byte) uint64 {
	if len(instruction) < 6 {
		return NO_BRANCH_DEST
	}

	inst0_15 := uint16(instruction[0])<<8 | uint16(instruction[1])
	inst16_31 := uint16(instruction[2])<<8 | uint16(instruction[3])
	inst32_47 := uint16(instruction[4])<<8 | uint16(instruction[5])
	opcode := instruction[0]

	var dest uint64 = NO_BRANCH_DEST

	// RR format branches
	if opcode == opBCR || opcode == opBALR || opcode == opBASR ||
		opcode == opBASSM || opcode == opBSM || opcode == opBCTR {

		r2 := int(inst0_15 & 0x0F)

		if opcode == opBCR {
			mask := int((inst0_15 >> 4) & 0x0F)
			if mask == 0 || r2 == 0 {
				return NO_BRANCH_DEST
			}
		} else {
			if r2 == 0 {
				return NO_BRANCH_DEST
			}
		}

		var regs PtraceRegs
		if err := PtraceGetRegs(pid, &regs); err == nil {
			if r2 >= 0 && r2 < 16 {
				regVal := regs.Gprs[r2]
				dest = regVal
			}
		}
	} else if opcode == opBC || opcode == opBAL || opcode == opBCT || opcode == opBAS {
		// RX format branches
		if opcode == opBC {
			mask := int((inst0_15 >> 4) & 0x0F)
			if mask == 0 {
				return NO_BRANCH_DEST
			}
		}

		index := int(inst0_15 & 0x0F)
		base := int((inst16_31 >> 12) & 0x0F)
		displacement := uint32(inst16_31 & 0x0FFF)

		var baseVal, indexVal uint64
		var regs PtraceRegs
		if err := PtraceGetRegs(pid, &regs); err == nil {
			if base != 0 && base < 16 {
				baseVal = regs.Gprs[base]
			}
			if index != 0 && index < 16 {
				indexVal = regs.Gprs[index]
			}
		}

		if base == 0 && index == 0 {
			dest = pc + uint64(displacement)
		} else {
			dest = baseVal + indexVal + uint64(displacement)
		}
	} else if opcode == opBXH || opcode == opBXLE {
		// RS format branches
		base := int((inst16_31 >> 12) & 0x0F)
		displacement := uint32(inst16_31 & 0x0FFF)

		var baseVal uint64
		var regs PtraceRegs
		if err := PtraceGetRegs(pid, &regs); err == nil {
			if base != 0 && base < 16 {
				baseVal = regs.Gprs[base]
			}
		}

		dest = baseVal + uint64(displacement)
	} else if opcode == opBA7 {
		// RI format branches
		subOp := int(inst0_15 & 0x0F)

		if subOp < 4 || subOp > 7 {
			return NO_BRANCH_DEST
		}

		if subOp == 0x04 {
			mask := int((inst0_15 >> 4) & 0x0F)
			if mask == 0 {
				return NO_BRANCH_DEST
			}
		}

		relDispl := int16(inst16_31)
		dest = uint64(int64(pc) + int64(relDispl)*2)
	} else if opcode == opBC0 {
		// RIL format branches
		subOp := int(inst0_15 & 0x0F)

		if subOp == 0x04 || subOp == 0x05 {
			if subOp == 0x04 {
				mask := int((inst0_15 >> 4) & 0x0F)
				if mask == 0 {
					return NO_BRANCH_DEST
				}
			}

			inst16_47 := (uint32(inst16_31) << 16) | uint32(inst32_47)
			relDispl4 := int32(inst16_47)
			dest = uint64(int64(pc) + int64(relDispl4)*2)
		}
	} else if opcode == opBEC {
		// RIEc format compare-and-branch
		subOp := instruction[4]

		isCompareBranch := (subOp >= 0x64 && subOp <= 0x67) ||
			(subOp >= 0x76 && subOp <= 0x77) ||
			(subOp >= 0x7C && subOp <= 0x7F)

		if isCompareBranch {
			ri4 := int16(inst16_31)
			dest = uint64(int64(pc) + int64(ri4)*2)
		}
	}

	if dest < 0x2000 {
		return NO_BRANCH_DEST
	}

	return dest
}

// setTempBreakpoint sets a temporary breakpoint at the specified address
func (s *singleStepState) setTempBreakpoint(pid int, addr uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Find free slot
	slot := -1
	for i := 0; i < MAX_TEMP_BREAKPOINTS; i++ {
		if !s.breakpoints[i].active {
			slot = i
			break
		}
	}

	if slot == -1 {
		return ENOMEM // All slots in use
	}

	// Read original instruction (up to 8 bytes for safety)
	_, err := PtracePeekText(pid, uintptr(addr), s.breakpoints[slot].originalInsn[:])
	if err != nil {
		return err
	}

	// Write SVC 144 (0x0A90) as breakpoint
	svc144 := []byte{0x0A, 0x90}
	_, err = PtracePokeText(pid, uintptr(addr), svc144)
	if err != nil {
		return err
	}

	s.breakpoints[slot].address = addr
	s.breakpoints[slot].active = true

	return nil
}

// removeTempBreakpoints removes all active temporary breakpoints
func (s *singleStepState) removeTempBreakpoints(pid int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var lastErr error
	for i := 0; i < MAX_TEMP_BREAKPOINTS; i++ {
		if !s.breakpoints[i].active {
			continue
		}

		// Restore original instruction (first 2 bytes are enough for SVC 144)
		_, err := PtracePokeText(pid, uintptr(s.breakpoints[i].address), s.breakpoints[i].originalInsn[:2])
		if err != nil {
			lastErr = err
		}

		s.breakpoints[i].active = false
	}

	return lastErr
}

// isTempBreakpoint checks if an address is a temporary breakpoint
func (s *singleStepState) isTempBreakpoint(addr uint64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	for i := 0; i < MAX_TEMP_BREAKPOINTS; i++ {
		if s.breakpoints[i].active && s.breakpoints[i].address == addr {
			return true
		}
	}
	return false
}

// PtraceSingleStep executes a single instruction using temporary breakpoints.
//
// This function:
// 1. Reads the current instruction at PSW
// 2. Calculates the next instruction address (sequential)
// 3. If branch instruction, calculates branch destination
// 4. Sets temporary breakpoints at both addresses
// 5. Continues execution until breakpoint hit
// 6. Restores original instructions and adjusts PSW
//
// Returns nil on success, error on failure.
func PtraceSingleStep(pid int) error {
	state := getProcessState(pid)

	// 1. Get current PSW
	var regs PtraceRegs
	if err := PtraceGetRegs(pid, &regs); err != nil {
		return err
	}
	pc := regs.Psw.Addr

	// 2. Read instruction at PC
	insn := make([]byte, 8)
	if _, err := PtracePeekText(pid, uintptr(pc), insn); err != nil {
		return err
	}

	// 3. Calculate next instruction addresses
	nextPC := GetNextInstructionAddr(pc, insn)
	branchDest := CalculateBranchDest(pid, pc, insn)

	// 4. Set temporary breakpoints
	if err := state.setTempBreakpoint(pid, nextPC); err != nil {
		return err
	}

	// If branch instruction, set breakpoint at branch destination too
	if branchDest != NO_BRANCH_DEST && branchDest != nextPC {
		if err := state.setTempBreakpoint(pid, branchDest); err != nil {
			state.removeTempBreakpoints(pid) // Cleanup first breakpoint
			return err
		}
	}

	// 5. Continue execution
	if err := PtraceCont(pid, 0); err != nil {
		state.removeTempBreakpoints(pid)
		return err
	}

	// 6. Wait for breakpoint hit (retry on EINTR)
	var status WaitStatus
	for {
		_, err := Wait4(pid, &status, 0, nil)
		if err == EINTR {
			continue // Retry on interrupted system call
		}
		if err != nil {
			state.removeTempBreakpoints(pid)
			return err
		}
		break
	}

	// Check if process stopped (should be at breakpoint)
	if !status.Stopped() {
		state.removeTempBreakpoints(pid)
		return ECHILD // Process exited or error
	}

	// 7. Get current PSW (should be at breakpoint + 2)
	if err := PtraceGetRegs(pid, &regs); err != nil {
		state.removeTempBreakpoints(pid)
		return err
	}

	// 8. Remove temporary breakpoints
	if err := state.removeTempBreakpoints(pid); err != nil {
		return err
	}

	// 9. Back up PSW by 2 bytes (to point at original instruction, not SVC 144)
	// Use direct PSW write like ztrace does, not full register set
	hitAddr := regs.Psw.Addr
	if state.isTempBreakpoint(hitAddr - 2) {
		if err := ptraceWritePSW(pid, uint32(hitAddr-2)); err != nil {
			return err
		}
	}

	return nil
}

// PtraceDetachWithCleanup detaches from a process and cleans up single-step state.
// Use this instead of PtraceDetach when using PtraceSingleStep.
func PtraceDetachWithCleanup(pid int) error {
	state := getProcessState(pid)
	state.removeTempBreakpoints(pid)
	cleanupProcessState(pid)
	return PtraceDetach(pid)
}
