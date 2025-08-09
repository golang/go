package gosym

import (
	"bytes"
	"debug/elf"
	"debug/macho"
	"encoding/binary"
	"fmt"
	"io"
)

func newMagic(r io.ReaderAt) (*Table, error) {
	magicBytes := make([]byte, 4)
	_, err := r.ReadAt(magicBytes, 0)
	if err != nil {
		return nil, err
	}
	if bytes.Equal(magicBytes, []byte(elf.ELFMAG)) {
		elf, err := elf.NewFile(r)
		if err != nil {
			return nil, err
		}
		return NewELF(elf)
	}
	magicLe := binary.LittleEndian.Uint32(magicBytes)
	magicBe := binary.BigEndian.Uint32(magicBytes)

	switch magicLe {
	case macho.Magic32, macho.Magic64, macho.MagicFat:
		macho, err := macho.NewFile(r)
		if err != nil {
			return nil, err
		}
		return NewMacho(macho)
	}
	switch magicBe {
	case macho.Magic32, macho.Magic64, macho.MagicFat:
		macho, err := macho.NewFile(r)
		if err != nil {
			return nil, err
		}
		return NewMacho(macho)
	}

	return nil, fmt.Errorf("%w: unrecognized magic: %x", ErrCorrupted, magicBytes)
}
