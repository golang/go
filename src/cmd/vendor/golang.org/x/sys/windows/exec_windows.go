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
//   - every back slash (\) is doubled, but only if immediately
//     followed by double quote (");
//   - every double quote (") is escaped by back slash (\);
//   - finally, s is wrapped with double quotes (arg -> "arg"),
//     but only if there is space or tab inside s.
func EscapeArg(s string) string {
	if len(s) == 0 {
		return `""`
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
		n += 2 // Reserve space for quotes.
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
	if len(args) == 0 {
		return ""
	}

	// Per https://learn.microsoft.com/en-us/windows/win32/api/shellapi/nf-shellapi-commandlinetoargvw:
	// “This function accepts command lines that contain a program name; the
	// program name can be enclosed in quotation marks or not.”
	//
	// Unfortunately, it provides no means of escaping interior quotation marks
	// within that program name, and we have no way to report them here.
	prog := args[0]
	mustQuote := len(prog) == 0
	for i := 0; i < len(prog); i++ {
		c := prog[i]
		if c <= ' ' || (c == '"' && i == 0) {
			// Force quotes for not only the ASCII space and tab as described in the
			// MSDN article, but also ASCII control characters.
			// The documentation for CommandLineToArgvW doesn't say what happens when
			// the first argument is not a valid program name, but it empirically
			// seems to drop unquoted control characters.
			mustQuote = true
			break
		}
	}
	var commandLine []byte
	if mustQuote {
		commandLine = make([]byte, 0, len(prog)+2)
		commandLine = append(commandLine, '"')
		for i := 0; i < len(prog); i++ {
			c := prog[i]
			if c == '"' {
				// This quote would interfere with our surrounding quotes.
				// We have no way to report an error, so just strip out
				// the offending character instead.
				continue
			}
			commandLine = append(commandLine, c)
		}
		commandLine = append(commandLine, '"')
	} else {
		if len(args) == 1 {
			// args[0] is a valid command line representing itself.
			// No need to allocate a new slice or string for it.
			return prog
		}
		commandLine = []byte(prog)
	}

	for _, arg := range args[1:] {
		commandLine = append(commandLine, ' ')
		// TODO(bcmills): since we're already appending to a slice, it would be nice
		// to avoid the intermediate allocations of EscapeArg.
		// Perhaps we can factor out an appendEscapedArg function.
		commandLine = append(commandLine, EscapeArg(arg)...)
	}
	return string(commandLine)
}

// DecomposeCommandLine breaks apart its argument command line into unescaped parts using CommandLineToArgv,
// as gathered from GetCommandLine, QUERY_SERVICE_CONFIG's BinaryPathName argument, or elsewhere that
// command lines are passed around.
// DecomposeCommandLine returns an error if commandLine contains NUL.
func DecomposeCommandLine(commandLine string) ([]string, error) {
	if len(commandLine) == 0 {
		return []string{}, nil
	}
	utf16CommandLine, err := UTF16FromString(commandLine)
	if err != nil {
		return nil, errorspkg.New("string with NUL passed to DecomposeCommandLine")
	}
	var argc int32
	argv, err := commandLineToArgv(&utf16CommandLine[0], &argc)
	if err != nil {
		return nil, err
	}
	defer LocalFree(Handle(unsafe.Pointer(argv)))

	var args []string
	for _, p := range unsafe.Slice(argv, argc) {
		args = append(args, UTF16PtrToString(p))
	}
	return args, nil
}

// CommandLineToArgv parses a Unicode command line string and sets
// argc to the number of parsed arguments.
//
// The returned memory should be freed using a single call to LocalFree.
//
// Note that although the return type of CommandLineToArgv indicates 8192
// entries of up to 8192 characters each, the actual count of parsed arguments
// may exceed 8192, and the documentation for CommandLineToArgvW does not mention
// any bound on the lengths of the individual argument strings.
// (See https://go.dev/issue/63236.)
func CommandLineToArgv(cmd *uint16, argc *int32) (argv *[8192]*[8192]uint16, err error) {
	argp, err := commandLineToArgv(cmd, argc)
	argv = (*[8192]*[8192]uint16)(unsafe.Pointer(argp))
	return argv, err
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
