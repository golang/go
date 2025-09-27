package gosym

import (
	"debug/macho"
	"encoding/binary"
	"fmt"
)

type machObject struct {
	*macho.File
}

func (o machObject) Endian() binary.ByteOrder {
	return o.ByteOrder
}

func (o machObject) Sections() (headers []SectionHeader, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%w: failed to parse section headers: %v", ErrCorrupted, r)
		}
	}()
	headers = make([]SectionHeader, 0, len(o.File.Sections))
	for _, s := range o.File.Sections {
		headers = append(headers, SectionHeader{
			Name: s.Name,
			Addr: s.Addr,
			Size: s.Size,
		})
	}
	return
}

func (o machObject) SectionData(i int8) (data []byte, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%w: failed to read section: %d: %v", ErrCorrupted, i, r)
		}
	}()
	data, err = o.File.Sections[i].Data()
	return
}
