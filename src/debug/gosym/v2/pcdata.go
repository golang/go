package gosym

import (
	"cmp"
	"encoding/binary"
	"fmt"
	"iter"
	"slices"
	"unique"
)

func (f *Function) endPc() (uint64, error) {
	// There is no direct way to get the end pc of a function.
	// We need to iterate one of pc sequences. Shortest should be the pcfile.
	t := f.table
	offset, err := t.funcdataPcfile(f.offset)
	if err != nil {
		return 0, err
	}
	pc, err := t.functabPc(f.idx)
	if err != nil {
		return 0, err
	}
	return t.lastPc(offset, pc)
}

// Parse inline tree.
func (f *Function) visitInlTree(
	entryPc uint64,
	includeSelf bool,
	visitor func(
		pcLo, pcHi uint64,
		funcNameOff, fileNo, startLine uint32,
		parentPC uint64,
	) error,
) error {
	t := f.table

	var selfNameOff uint32
	var selfStartLine uint32
	if includeSelf {
		var err error
		selfNameOff, err = t.funcdataNameOff(f.offset)
		if err != nil {
			return err
		}
		selfStartLine, err = t.funcdataStartLine(f.offset)
		if err != nil {
			return err
		}
	}

	offset, err := t.funcdataPcfile(f.offset)
	if err != nil {
		return err
	}
	pcFileIter, err := t.iterPCValues(offset, entryPc)
	if err != nil {
		return err
	}
	nextFile, stopFile := iter.Pull(pcFileIter)
	defer stopFile()

	inlTree, err := t.funcdataInlTree(f.offset)
	if err != nil {
		return err
	}
	if inlTree == nil {
		// This function has no inline ranges.
		if includeSelf {
			file, ok := nextFile()
			if !ok {
				return fmt.Errorf("%w: missing file entry for pc %x", ErrCorrupted, entryPc)
			}
			return visitor(entryPc, file.PC, selfNameOff, uint32(file.Val), selfStartLine, 0)
		}
		return nil
	}

	offset, err = t.funcdataInlTreeIndex(f.offset)
	if err != nil {
		return err
	}
	pcValueSeq, err := t.iterPCValues(offset, entryPc)
	if err != nil {
		return err
	}
	pc := entryPc
	curFile := pcValue{
		PC:  0,
		Val: -1,
	}
	for inlFuncIdx := range pcValueSeq {
		// [pc, inlFuncIdx.PC) maps to inline function data at inlFuncIdx.Val.
		var funcNameOff uint32
		var startLine uint32
		var parentPC uint64
		if inlFuncIdx.Val < 0 {
			// This range doesn't correspond to any inline function, it belongs to f itself.
			if !includeSelf {
				pc = inlFuncIdx.PC
				continue
			}
			funcNameOff = selfNameOff
			startLine = selfStartLine
		} else {
			call, err := readInlTree(inlTree, uint32(inlFuncIdx.Val))
			if err != nil {
				return err
			}
			if call.funcID == t.wrapperFuncID {
				pc = inlFuncIdx.PC
				continue
			}
			funcNameOff = call.nameOff
			startLine = call.startLine
			parentPC = entryPc + uint64(call.parentPC)
		}
		// Find the file that covers the beginning of the current PC range.
		for pc >= curFile.PC {
			var ok bool
			curFile, ok = nextFile()
			if !ok {
				return fmt.Errorf("%w: missing file entry for pc %x", ErrCorrupted, pc)
			}
		}
		err := visitor(pc, inlFuncIdx.PC, funcNameOff, uint32(curFile.Val), startLine, parentPC)
		if err != nil {
			return err
		}
		pc = inlFuncIdx.PC
	}
	return nil
}

// Parse inline tree to extract list of inline functions.
func (f *Function) inlines(buf []InlineFunction) ([]InlineFunction, error) {
	t := f.table
	entryPc, err := t.functabPc(f.idx)
	if err != nil {
		return buf, err
	}
	err = f.visitInlTree(entryPc, false, /*includeSelf*/
		func(_, _ uint64, funcNameOff, fileNo, startLine uint32, _ uint64) error {
			funcName := t.funcName(funcNameOff)
			offset, err := t.funcdataFileNoToOff(f.offset, fileNo)
			if err != nil {
				return err
			}
			file := t.fileName(offset)
			buf = append(buf, InlineFunction{
				Name:      funcName,
				File:      file,
				StartLine: startLine,
			})
			return nil
		})
	if err != nil {
		return nil, err
	}
	slices.SortFunc(buf, func(a, b InlineFunction) int {
		return cmp.Compare(a.Name.Value(), b.Name.Value())
	})
	buf = slices.CompactFunc(buf, func(a, b InlineFunction) bool {
		return a.Name.Value() == b.Name.Value()
	})
	return buf, nil
}

type inlineRange struct {
	pcLo, pcHi  uint64
	funcNameOff uint32
	fileNo      uint32
	parentPC    uint64
}

// Parse inline tree lookup table for function f, to produce list of its
// pc subranges mapped to either some inline function or f itself. Returned
// ranges are sorted by pcLo and non-overlapping. Ranges corresponding to
// wrapper functions are skipped. If the function has no inline functions,
// returns nil.
func (f *Function) inlineMapping(entryPc uint64, buf []inlineRange) ([]inlineRange, error) {
	err := f.visitInlTree(entryPc, true, /*includeSelf*/
		func(pcLo, pcHi uint64, funcNameOff uint32, fileNo uint32, _ uint32, parentPC uint64) error {
			buf = append(buf, inlineRange{
				pcLo:        pcLo,
				pcHi:        pcHi,
				funcNameOff: funcNameOff,
				fileNo:      fileNo,
				parentPC:    parentPC,
			})
			return nil
		})
	if err != nil {
		return nil, err
	}
	return buf, nil
}

type inlinedCall struct {
	funcID    uint8
	nameOff   uint32
	parentPC  uint32
	startLine uint32
}

func readInlTree(inlineTree []byte, fIdx uint32) (inlinedCall, error) {
	const (
		inlinedCallSize       = 16
		funcIDOffset          = 0
		functionNameOffOffset = 4
		parentPCOffset        = 8
		startLineOffset       = 12
	)
	offset := inlinedCallSize * int(fIdx)
	if offset+inlinedCallSize > len(inlineTree) {
		return inlinedCall{}, fmt.Errorf("%w: inl tree out of bounds, off=%d", ErrCorrupted, offset)
	}
	return inlinedCall{
		funcID:    inlineTree[offset+funcIDOffset],
		nameOff:   binary.LittleEndian.Uint32(inlineTree[offset+functionNameOffOffset:]),
		parentPC:  binary.LittleEndian.Uint32(inlineTree[offset+parentPCOffset:]),
		startLine: binary.LittleEndian.Uint32(inlineTree[offset+startLineOffset:]),
	}, nil
}

type linesCache struct {
	inlines []inlineRange
	funcs   map[uint32]unique.Handle[string]
	files   map[uint32]unique.Handle[string]
}

func (f *Function) lines(buf LinesResult) (LinesResult, error) {
	t := f.table

	if buf.linesCache.funcs == nil {
		buf.linesCache.funcs = make(map[uint32]unique.Handle[string])
	}
	if buf.linesCache.files == nil {
		buf.linesCache.files = make(map[uint32]unique.Handle[string])
	}

	entryPc, err := t.functabPc(f.idx)
	if err != nil {
		return buf, err
	}

	// Calculate mapping from pc ranges to functions.
	defer clear(buf.linesCache.inlines)
	buf.linesCache.inlines, err = f.inlineMapping(entryPc, buf.linesCache.inlines)
	if err != nil {
		return buf, err
	}
	funcs := buf.linesCache.inlines
	funcIdx := 0

	clear(buf.FunctionLines)
	defer clear(buf.linesCache.funcs)
	defer clear(buf.linesCache.files)
	upsert := func(
		pcLo, pcHi uint64,
		funcNameOff, fileNo, line uint32,
		parentPC uint64,
	) error {
		var funcName unique.Handle[string]
		var ok bool
		if funcName, ok = buf.linesCache.funcs[funcNameOff]; !ok {
			funcName = t.funcName(funcNameOff)
			buf.linesCache.funcs[funcNameOff] = funcName
		}
		var file unique.Handle[string]
		if file, ok = buf.linesCache.files[fileNo]; !ok {
			offset, err := t.funcdataFileNoToOff(f.offset, fileNo)
			if err != nil {
				return err
			}
			file = t.fileName(offset)
			buf.linesCache.files[fileNo] = file
		}
		buf.FunctionLines = append(buf.FunctionLines, FunctionLine{
			PCLo:     pcLo,
			PCHi:     pcHi,
			Name:     funcName,
			File:     file,
			Line:     line,
			ParentPC: parentPC,
		})
		return nil
	}

	// Join function mapping with line mapping. This mapping can be many-to-many.
	offset, err := t.funcdataPcln(f.offset)
	if err != nil {
		return buf, err
	}
	pcValueSeq, err := t.iterPCValues(offset, entryPc)
	if err != nil {
		return buf, err
	}
	pc := entryPc
	for line := range pcValueSeq {
		// [pc, line.PC) maps to line.Val.
		// Skip function ranges that end before current line range.
		for funcIdx < len(funcs) && pc >= funcs[funcIdx].pcHi {
			funcIdx++
		}
		// Iterate over function ranges that end within the current line range.
		for funcIdx < len(funcs) && line.PC >= funcs[funcIdx].pcHi {
			err := upsert(
				max(pc, funcs[funcIdx].pcLo),
				funcs[funcIdx].pcHi,
				funcs[funcIdx].funcNameOff,
				funcs[funcIdx].fileNo,
				uint32(line.Val),
				funcs[funcIdx].parentPC,
			)
			if err != nil {
				return buf, nil
			}
			funcIdx++
		}
		// Check if the next function range starts within the current line range.
		if funcIdx < len(funcs) && pc >= funcs[funcIdx].pcLo {
			err := upsert(
				max(pc, funcs[funcIdx].pcLo),
				line.PC,
				funcs[funcIdx].funcNameOff,
				funcs[funcIdx].fileNo,
				uint32(line.Val),
				funcs[funcIdx].parentPC,
			)
			if err != nil {
				return buf, nil
			}
		}
		pc = line.PC
	}
	return buf, nil
}

func (f *Function) resolveLocations(pc uint64, buf []Location) ([]Location, error) {
	t := f.table

	entryPc, err := t.functabPc(f.idx)
	if err != nil {
		return buf, err
	}
	pcfile, err := t.funcdataPcfile(f.offset)
	if err != nil {
		return buf, err
	}
	pcln, err := t.funcdataPcln(f.offset)
	if err != nil {
		return buf, err
	}

	addLocation := func(nameOff uint32, pc uint64) error {
		fno, err := t.pcValue(pcfile, entryPc, pc)
		if err != nil {
			return err
		}
		foff, err := t.funcdataFileNoToOff(f.offset, uint32(fno))
		if err != nil {
			return err
		}
		line, err := t.pcValue(pcln, entryPc, pc)
		if err != nil {
			return err
		}
		buf = append(buf, Location{
			Function: t.funcName(nameOff),
			File:     t.fileName(foff),
			Line:     uint32(line),
		})
		return nil
	}

	inlTree, err := t.funcdataInlTree(f.offset)
	if err != nil {
		return buf, err
	}
	if inlTree != nil {
		offset, err := t.funcdataInlTreeIndex(f.offset)
		if err != nil {
			return buf, err
		}
		for {
			inlIdx, err := t.pcValue(offset, entryPc, pc)
			if err != nil {
				return buf, fmt.Errorf("%w: pc outside of any function", ErrPcNotFound)
			}
			if inlIdx < 0 {
				break
			}
			call, err := readInlTree(inlTree, uint32(inlIdx))
			if err != nil {
				return buf, err
			}
			if call.funcID != t.wrapperFuncID {
				err := addLocation(call.nameOff, pc)
				if err != nil {
					return buf, err
				}
			}
			pc = entryPc + uint64(call.parentPC)
		}
	}
	nameOff, err := t.funcdataNameOff(f.offset)
	if err != nil {
		return buf, err
	}
	err = addLocation(nameOff, pc)
	if err != nil {
		return buf, err
	}
	return buf, nil
}
