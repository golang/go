// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fork, exec, wait, etc.

package windows

import (
	errorspkg "errors"
	"unsafe"
)

// EscapeArg rewrites command line argument s as prescribed
// in http://msdn.microsoft.com/en-us/library/ms880421.
// This function returns "" (2 double quotes) if s is empty.
// Alternatively, these transformations are done:
// - every back slash (\) is doubled, but only if immediately
//   followed by double quote (");
// - every double quote (") is escaped by back slash (\);
// - finally, s is wrapped with double quotes (arg -> "arg"),
//   but only if there is space or tab inside s.
func EscapeArg(s string) string {
	if len(s) == 0 {
		return "\"\""
	}
	n := len(s)
	hasSpace := false
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"', '\\':
			n++
		case ' ', '\t':
			hasSpace = true
		}
	}
	if hasSpace {
		n += 2
	}
	if n == len(s) {
		return s
	}

	qs := make([]byte, n)
	j := 0
	if hasSpace {
		qs[j] = '"'
		j++
	}
	slashes := 0
	for i := 0; i < len(s); i++ {
		switch s[i] {
		default:
			slashes = 0
			qs[j] = s[i]
		case '\\':
			slashes++
			qs[j] = s[i]
		case '"':
			for ; slashes > 0; slashes-- {
				qs[j] = '\\'
				j++
			}
			qs[j] = '\\'
			j++
			qs[j] = s[i]
		}
		j++
	}
	if hasSpace {
		for ; slashes > 0; slashes-- {
			qs[j] = '\\'
			j++
		}
		qs[j] = '"'
		j++
	}
	return string(qs[:j])
}

// ComposeCommandLine escapes and joins the given arguments suitable for use as a Windows command line,
// in CreateProcess's CommandLine argument, CreateService/ChangeServiceConfig's BinaryPathName argument,
// or any program that uses CommandLineToArgv.
func ComposeCommandLine(args []string) string {
	var commandLine string
	for i := range args {
		if i > 0 {
			commandLine += " "
		}
		commandLine += EscapeArg(args[i])
	}
	return commandLine
}

// DecomposeCommandLine breaks apart its argument command line into unescaped parts using CommandLineToArgv,
// as gathered from GetCommandLine, QUERY_SERVICE_CONFIG's BinaryPathName argument, or elsewhere that
// command lines are passed around.
func DecomposeCommandLine(commandLine string) ([]string, error) {
	if len(commandLine) == 0 {
		return []string{}, nil
	}
	var argc int32
	argv, err := CommandLineToArgv(StringToUTF16Ptr(commandLine), &argc)
	if err != nil {
		return nil, err
	}
	defer LocalFree(Handle(unsafe.Pointer(argv)))
	var args []string
	for _, v := range (*argv)[:argc] {
		args = append(args, UTF16ToString((*v)[:]))
	}
	return args, nil
}

func CloseOnExec(fd Handle) {
	SetHandleInformation(Handle(fd), HANDLE_FLAG_INHERIT, 0)
}

// FullPath retrieves the full path of the specified file.
func FullPath(name string) (path string, err error) {
	p, err := UTF16PtrFromString(name)
	if err != nil {
		return "", err
	}
	n := uint32(100)
	for {
		buf := make([]uint16, n)
		n, err = GetFullPathName(p, uint32(len(buf)), &buf[0], nil)
		if err != nil {
			return "", err
		}
		if n <= uint32(len(buf)) {
			return UTF16ToString(buf[:n]), nil
		}
	}
}

// NewProcThreadAttributeList allocates a new ProcThreadAttributeListContainer, with the requested maximum number of attributes.
func NewProcThreadAttributeList(maxAttrCount uint32) (*ProcThreadAttributeListContainer, error) {
	var size uintptr
	err := initializeProcThreadAttributeList(nil, maxAttrCount, 0, &size)
	if err != ERROR_INSUFFICIENT_BUFFER {
		if err == nil {
			return nil, errorspkg.New("unable to query buffer size from InitializeProcThreadAttributeList")
		}
		return nil, err
	}
	alloc, err := LocalAlloc(LMEM_FIXED, uint32(size))
	if err != nil {
		return nil, err
	}
	// size is guaranteed to be ≥1 by InitializeProcThreadAttributeList.
	al := &ProcThreadAttributeListContainer{data: (*ProcThreadAttributeList)(unsafe.Pointer(alloc))}
	err = initializeProcThreadAttributeList(al.data, maxAttrCount, 0, &size)
	if err != nil {
		return nil, err
	}
	return al, err
}

// Update modifies the ProcThreadAttributeList using UpdateProcThreadAttribute.
func (al *ProcThreadAttributeListContainer) Update(attribute uintptr, value unsafe.Pointer, size uintptr) error {
	al.pointers = append(al.pointers, value)
	return updateProcThreadAttribute(al.data, 0, attribute, value, size, nil, nil)
}

// Delete frees ProcThreadAttributeList's resources.
func (al *ProcThreadAttributeListContainer) Delete() {
	deleteProcThreadAttributeList(al.data)
	LocalFree(Handle(unsafe.Pointer(al.data)))
	al.data = nil
	al.pointers = nil
}

// List returns the actual ProcThreadAttributeList to be passed to StartupInfoEx.
func (al *ProcThreadAttributeListContainer) List() *ProcThreadAttributeList {
	return al.data
}
